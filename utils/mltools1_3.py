import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import os
import json

def plot_confusion_matrix(y_true, y_pred, title, classes, save_filename=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    

    plt.rcParams.update({
        'font.size': 14,          
        'axes.titlesize': 16,     
        'axes.labelsize': 15      
    })
    
    plt.figure(figsize=(12, 10), dpi=300)  
    
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2%', 
                cmap='Blues',
                xticklabels=classes, 
                yticklabels=classes,
                annot_kws={"size": 13},  
                cbar_kws={"shrink": 0.75})  
    
    plt.title(title, pad=20)  
    plt.ylabel('True Label', labelpad=10)  
    plt.xlabel('Predicted Label', labelpad=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_filename:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, 
                   bbox_inches='tight', 
                   dpi=300,  
                   transparent=False)
    plt.close()

def plot_snr_confusion_matrix(model, X, Y_true, SNRs, classes, title_prefix, save_dir='figure'):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'figure.dpi': 300  
    })

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

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8), dpi=300)  
        ax = sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=classes, 
            yticklabels=classes,
            annot_kws={"size": 12}, 
            cbar_kws={"shrink": 0.75}
        )
        
        ax.set_title(f"{title_prefix} (SNR={snr}dB, Acc={accuracy:.2f})", pad=15)
        ax.set_xlabel('Predicted Label', labelpad=10)
        ax.set_ylabel('True Label', labelpad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{title_prefix.replace(' ', '_')}_SNR_{snr}.png"
        plt.savefig(
            filename, 
            bbox_inches='tight', 
            dpi=300,  
            transparent=False
        )
        plt.close()

def calculate_confusion_matrix(Y_true, Y_pred, classes):
    cm = confusion_matrix(Y_true, Y_pred)
    confnorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return confnorm, np.diag(cm).sum(), cm.sum() - np.diag(cm).sum()



def evaluate_and_save_accuracy_json(model, X, Y_mod_true, Y_sig_true, SNRs, config, save_path='results/accuracy_results.json', device='cpu'):
    unique_snrs = sorted(np.unique(SNRs))
    snr_mod_acc = {}
    snr_sig_acc = {}

    for snr in unique_snrs:
        mask = (SNRs == snr)
        X_snr = X[mask]
        Y_mod_snr = Y_mod_true[mask]
        Y_sig_snr = Y_sig_true[mask]

        with torch.no_grad():
            if isinstance(X_snr, np.ndarray):
                X_snr = torch.tensor(X_snr, dtype=torch.float32).to(device)
            outputs = model(X_snr)
            mod_out, sig_out = outputs

            mod_pred = torch.argmax(mod_out, dim=1).cpu().numpy()
            sig_pred = torch.argmax(sig_out, dim=1).cpu().numpy()

        mod_acc = float(np.mean(mod_pred == Y_mod_snr))
        sig_acc = float(np.mean(sig_pred == Y_sig_snr))

        snr_mod_acc[str(snr)] = round(mod_acc, 4)
        snr_sig_acc[str(snr)] = round(sig_acc, 4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result = {
        "config": config,
        "snrs": unique_snrs,
        "snr_modulation_acc": snr_mod_acc,
        "snr_signal_acc": snr_sig_acc
    }

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)