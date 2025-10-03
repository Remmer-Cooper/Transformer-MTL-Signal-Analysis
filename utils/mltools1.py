import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, title, classes, save_filename=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_filename:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, bbox_inches='tight')
    plt.close()

def plot_snr_confusion_matrix(model, X, Y_true, SNRs, classes, title_prefix, save_dir='figure'):
    unique_snrs = np.unique(SNRs)
    for snr in unique_snrs:
        mask = (SNRs == snr)
        X_snr = X[mask]
        Y_snr = Y_true[mask]
        
        with torch.no_grad():
            if isinstance(X_snr, np.ndarray):
                X_snr = torch.tensor(X_snr, dtype=torch.float32)
            outputs = model(X_snr)
            if isinstance(outputs, tuple): 
                outputs = outputs[0] if "Modulation" in title_prefix else outputs[1]
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        
        y_true = np.argmax(Y_snr, axis=1) if len(Y_snr.shape) > 1 else Y_snr
        
        accuracy = np.mean(y_true == y_pred)
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title(f"{title_prefix} (SNR={snr}dB, Acc={accuracy:.2f})")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{title_prefix.replace(' ', '_')}_SNR_{snr}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

def calculate_confusion_matrix(Y_true, Y_pred, classes):
    cm = confusion_matrix(Y_true, Y_pred)
    confnorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return confnorm, np.diag(cm).sum(), cm.sum() - np.diag(cm).sum()