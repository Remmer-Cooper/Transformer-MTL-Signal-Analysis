import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
from tqdm import tqdm
import psutil  
import time


def load_data(file_path):
    """ Keep this function as is, it is used in multiple scripts """
    X = []
    Y_mod = []
    Y_sig = []
    SNRs = []
    
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data = f[key][:]
            X.append(data)
            parts = key.strip("()").split(", ")
            mod_label = parts[0].strip("'")
            sig_label = parts[1].strip("'")
            snr = float(parts[2].strip("'"))
            Y_mod.append(mod_label)
            Y_sig.append(sig_label)
            SNRs.append(snr)
    
    X = np.array(X)
    Y_mod = np.array(Y_mod)
    Y_sig = np.array(Y_sig)
    SNRs = np.array(SNRs)
    
    X_complex = X[:, :128] + 1j * X[:, 128:]
    
    mod_encoder = LabelEncoder()
    sig_encoder = LabelEncoder()
    Y_mod_encoded = mod_encoder.fit_transform(Y_mod)
    Y_sig_encoded = sig_encoder.fit_transform(Y_sig)
    
    return X_complex, Y_mod_encoded, Y_sig_encoded, mod_encoder, sig_encoder, SNRs


def generate_constellation(iq):
    
    # normalize IQ data to [-1, 1]
    points = np.column_stack([
        ((iq.real + 1) * 7.5).clip(0, 15).astype(int),
        ((iq.imag + 1) * 7.5).clip(0, 15).astype(int)
    ])
    
    img = np.zeros((16, 16), dtype=np.float32)
    for x, y in points:
        img[y, x] += 1
    
    # normalize to [0, 1]
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img, (3,3), 0.5)
    return img



def preprocess_data(X):
    """preprocess the data to generate constellation images and reshape for model input"""
    X_combined = np.zeros((len(X), 3, 16, 16), dtype=np.float32)
    
    for i in tqdm(range(len(X)), desc="Preprocessing data"):
        #Prepare IQ data
        iq = np.stack([X[i].real, X[i].imag])  # [2, 128]
        iq = iq.reshape(2, 8, 16)             # [2, 8, 16]
        iq = np.repeat(iq, 2, axis=1)         # [2, 16, 16]
        
        # Generate constellation image
        const = generate_constellation(X[i])
        
        # Combine IQ and constellation image
        X_combined[i] = np.concatenate([
            iq,                  # [2,16,16]
            const[np.newaxis,...] # [1,16,16]
        ], axis=0)
    
    return X_combined
def split_data(X, Y_mod, Y_sig, SNRs, train_ratio=0.7, val_ratio=0.2):
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)
    
    X_train = X[indices[:train_end]]
    Y_mod_train = Y_mod[indices[:train_end]]
    Y_sig_train = Y_sig[indices[:train_end]]
    SNRs_train = SNRs[indices[:train_end]]
    
    X_val = X[indices[train_end:val_end]]
    Y_mod_val = Y_mod[indices[train_end:val_end]]
    Y_sig_val = Y_sig[indices[train_end:val_end]]
    SNRs_val = SNRs[indices[train_end:val_end]]
    
    X_test = X[indices[val_end:]]
    Y_mod_test = Y_mod[indices[val_end:]]
    Y_sig_test = Y_sig[indices[val_end:]]
    SNRs_test = SNRs[indices[val_end:]]
    
    return (X_train, Y_mod_train, Y_sig_train, SNRs_train), \
           (X_val, Y_mod_val, Y_sig_val, SNRs_val), \
           (X_test, Y_mod_test, Y_sig_test, SNRs_test)