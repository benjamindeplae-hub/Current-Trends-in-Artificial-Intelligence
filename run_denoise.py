import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate
from tensorflow.keras.models import Model

# 1. LOAD DATA
def load_data():
    eeg = np.load('data/EEG_all_epochs.npy')
    eog = np.load('data/EOG_all_epochs.npy')
    emg = np.load('data/EMG_all_epochs.npy')
    
    return eeg, eog, emg

# 3. EXECUTE
eeg, eog, emg = load_data()
print("EEG shape:", eeg.shape)
print("EOG shape:", eog.shape)
print("EMG shape:", emg.shape)