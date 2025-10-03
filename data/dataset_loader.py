
import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
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

def preprocess_data(X):
    X_real = np.stack([X.real, X.imag], axis=-1)
    X_reshaped = X_real.reshape(-1, 16, 16, 1)
    X_reshaped = np.transpose(X_reshaped, (0, 3, 1, 2))  # [B, C, H, W]
    return X_reshaped

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