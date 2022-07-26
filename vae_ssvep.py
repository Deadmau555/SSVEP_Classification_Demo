import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold

from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.losses import categorical_crossentropy

from numpy import matlib as mb

import utils.cca_utils as su
import utils.vae_utils as va

import torch
import tensorflow as tf
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
tf.config.experimental_run_functions_eagerly(True)


np.random.seed(42)
torch.manual_seed(42)

data_path = os.path.abspath('data')
all_segment_data = dict()
window_len = 1
shift_len = 1
sample_rate = 256
batch_size = 16
num_epochs = 10
duration = int(window_len*sample_rate)
flicker_freq = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 
                       10.25, 12.25, 14.25, 10.75, 12.75, 14.75])


CNN_PARAMS = {
    'batch_size': 64,
    'epochs': 50,
    'droprate': 0.25,
    'learning_rate': 0.001,
    'lr_decay': 0.0,
    'l2_lambda': 0.0001,
    'momentum': 0.9,
    'kernel_f': 10,
    'n_ch': 8,
    'num_classes': 12}


def get_CNN_data(features_data, CNN_PARAMS):
    features_data = features_data.view(features_data.shape[4], features_data.shape[1], features_data.shape[0], -1)
    total_epochs_per_class = features_data.shape[3]
    
    train_data = features_data.view(-1, features_data.shape[1], features_data.shape[0])
    features_data = []
    
    class_labels = torch.arange(CNN_PARAMS['num_classes'])
    labels = (mb.repmat(class_labels, total_epochs_per_class, 1).T).ravel()
    labels = to_categorical(labels)
    return train_data, labels



def get_vae_data(all_segment_data, CNN_PARAMS):
    for subject in range(0,CNN_PARAMS["kernel_f"]):
        features_data = torch.from_numpy(all_segment_data[f's{subject+1}'])
        features_data = features_data.view(features_data.shape[4], features_data.shape[1], features_data.shape[0], -1)
        total_epochs_per_class = features_data.shape[3]
        
        train_data = features_data.view(-1, features_data.shape[1], features_data.shape[0])
        features_data = []
        
        class_labels = torch.arange(CNN_PARAMS['num_classes'])
        labels = (mb.repmat(class_labels, total_epochs_per_class, 1).T).ravel()
        labels = to_categorical(labels)
        if subject == 0:
            all_data = train_data
            all_labels = torch.from_numpy(labels)
        else:
            all_data = torch.cat((all_data, train_data), dim=0)
            all_labels = torch.cat((all_labels, torch.from_numpy(labels)), dim=0)
    return all_data, all_labels


def get_segment_data(CNN_PARAMS, sample_rate = 256, window_len = 1, shift_len = 1):
    # process all egg data
    for subject in tqdm(range(0, CNN_PARAMS["kernel_f"])):
        dataset = sio.loadmat(f'data/s{subject+1}.mat')
        eeg = np.array(dataset['eeg'], dtype='float32')
        
        num_classes = eeg.shape[0]
        n_ch = eeg.shape[1]
        total_trial_len = eeg.shape[2]
        num_trials = eeg.shape[3]
        sample_rate = 256

        filtered_data = su.get_filtered_eeg(eeg, 6, 80, 4, sample_rate)
        all_segment_data[f's{subject+1}'] = su.get_segmented_epochs(filtered_data, window_len, 
                                                            shift_len, sample_rate)
    return all_segment_data


# initialize the vae model
vae_model = va.EEG_CNN_VAE()
vae_model.apply(va.weights_init) 
vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.000003)   

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# get processed data
all_segment_data = get_segment_data(CNN_PARAMS, sample_rate)
# prepare the data needed by vae 
vae_data, vae_labels = get_vae_data(all_segment_data, CNN_PARAMS)

vae_dataset = torch.utils.data.TensorDataset(vae_data, vae_labels)
vae_dataloader = torch.utils.data.DataLoader(dataset=vae_dataset, batch_size=batch_size, shuffle=True)


vae_model.train()
for epoch in range(num_epochs):
    for i, data in enumerate(vae_dataloader, 0):
        data, labels = data
        data = data.type(Tensor)
        
        vae_optimizer.zero_grad()
        recon_data, mu, logvar = vae_model(data)   
        vae_loss, bce, kld = va.loss_fn(recon_data, data, mu, logvar)

        vae_loss.backward()
        vae_optimizer.step()

    to_print = "Epoch[{}/{}] Loss: {:.6f} {:.6f} {:.6f}".format(epoch+1, num_epochs, vae_loss.item(), bce.item(), kld.item())
    print(to_print)

# Generating new data
with torch.no_grad():
    vae_model.eval()

    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True)

    all_acc = np.zeros((10, 1))
    
    for subject in range(0,10):
        features_data = torch.from_numpy(all_segment_data[f's{subject+1}'])
        cnn_data, cnn_labels = get_CNN_data(features_data, CNN_PARAMS)
        kf.get_n_splits(cnn_data)
        cv_acc = np.zeros((num_folds, 1))
        fold = -1
    
        for train_index, test_index in kf.split(cnn_data):
            x_tr, x_ts = cnn_data[train_index], cnn_data[test_index]
            y_tr, y_ts = cnn_labels[train_index], cnn_labels[test_index]
        
            hx_tr = vae_model.hidden_encode(x_tr.type(Tensor))
            hx_ts = vae_model.hidden_encode(x_ts.type(Tensor))
            
            vhx_tr = hx_tr.view(hx_tr.shape[0], hx_tr.shape[1], hx_tr.shape[2], 1)
            vhx_ts = hx_ts.view(hx_ts.shape[0], hx_ts.shape[1], hx_ts.shape[2], 1)

            input_shape = np.array([vhx_tr.shape[1], vhx_tr.shape[2], vhx_tr.shape[3]])


            fold = fold + 1
            print("Subject:", subject+1, "Fold:", fold+1, "Training...")
            
            cnn_model = su.CNN_model(input_shape, CNN_PARAMS)
            
            sgd = optimizers.SGD(lr=CNN_PARAMS['learning_rate'], decay=CNN_PARAMS['lr_decay'], 
                                momentum=CNN_PARAMS['momentum'], nesterov=False)
            cnn_model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=["accuracy"])
            # history = cnn_model.fit(x_tr, y_tr, batch_size=CNN_PARAMS['batch_size'], 
            #                     epochs=CNN_PARAMS['epochs'], verbose=0)
            # x_tr = x_tr.unsqueeze(dim=-1).numpy()
            # x_ts = x_ts.unsqueeze(dim=-1).numpy()
            history = cnn_model.fit(vhx_tr.numpy(), y_tr, batch_size=CNN_PARAMS['batch_size'], 
                                epochs=CNN_PARAMS['epochs'], verbose=0)

            score = cnn_model.evaluate(vhx_ts.numpy(), y_ts, verbose=0) 
            cv_acc[fold, :] = score[1]*100
            print("%s: %.2f%%" % (cnn_model.metrics_names[1], score[1]*100))
        
        all_acc[subject] = np.mean(cv_acc)
        print("...................................................")
        print("Subject:", subject+1, " - Accuracy:", all_acc[subject],"%")
        print("...................................................")
print(".....................................................................................")
print("Overall Accuracy Across Subjects:", np.mean(all_acc), "%", "std:", np.std(all_acc), "%")
print(".....................................................................................")
