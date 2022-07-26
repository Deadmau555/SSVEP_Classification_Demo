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

import warnings
warnings.filterwarnings('ignore')

data_path = os.path.abspath('data')
all_segment_data = dict()
window_len = 1
shift_len = 1
sample_rate = 256
num_epochs = 100
duration = int(window_len*sample_rate)
flicker_freq = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 
                       10.25, 12.25, 14.25, 10.75, 12.75, 14.75])

# process all egg data
for subject in range(0, 10):
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
print(123)

model = va.EEG_CNN_VAE()
model.apply(va.weights_init) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

for num_class in range(0,12):
    for subject in range(0,10):
        temp_data = torch.from_numpy(all_segment_data[f's{subject+1}'][num_class])
        temp_data = temp_data.view(temp_data.shape[0], temp_data.shape[1], -1)
        if subject == 0: 
            class_data = temp_data
            continue
        # temp_data = all_segment_eeg[f's{subject+1}'][num_class]
        class_data = torch.cat((class_data, temp_data))
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        recon_data, mu, logvar = model(class_data.type(Tensor))   
        loss, bce, kld = va.loss_fn(recon_data, class_data.type(Tensor), mu, logvar)

        loss.backward()
        optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.6f} {:.6f} {:.6f}".format(epoch+1, num_epochs, loss.item(), bce.item(), kld.item())
        print(to_print)

    # Generating new data
    new_data = []
    num_data_to_generate = 500
    with torch.no_grad():
        model.eval()
        for epoch in range(num_data_to_generate):

            z = torch.randn(1, 16)
            recon_data = model.decode(z).cpu().numpy()

            new_data.append(recon_data)

        new_data = np.concatenate(new_data) 
        new_data = np.asarray(new_data)
        print(123)

