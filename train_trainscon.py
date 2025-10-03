import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from models.build_mtl_modelcon import VisionTransformerMTL
from data.dataset_loader_con import load_data, preprocess_data, split_data
from utils.mltools1_1 import plot_confusion_matrix, calculate_confusion_matrix, plot_snr_confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X, Y_mod, Y_sig, mod_encoder, sig_encoder, SNRs = load_data(r'D:\Test\mywork\Muti_task_classify\datas\RadComDynamic\RadComDynamic.hdf5')
X_processed = preprocess_data(X)
print(f"Processed shape: {X_processed.shape}")  # should be [N,3,16,16]

indices = np.random.permutation(len(X_processed))
X_processed = X_processed[indices]
Y_mod = Y_mod[indices]
Y_sig = Y_sig[indices]
SNRs = SNRs[indices]


(X_train, Y_mod_train, Y_sig_train, SNRs_train), \
(X_val, Y_mod_val, Y_sig_val, SNRs_val), \
(X_test, Y_mod_test, Y_sig_test, SNRs_test) = split_data(X_processed, Y_mod, Y_sig, SNRs)

class RadioDataset(Dataset):
    def __init__(self, X, Y_mod, Y_sig):
        self.X = X
        self.Y_mod = Y_mod
        self.Y_sig = Y_sig

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y_mod[idx], dtype=torch.long),
            torch.tensor(self.Y_sig[idx], dtype=torch.long)
        )

    def __len__(self):
        return len(self.X)

train_dataset = RadioDataset(X_train, Y_mod_train, Y_sig_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = RadioDataset(X_val, Y_mod_val, Y_sig_val)
val_loader = DataLoader(val_dataset, batch_size=32)

model = VisionTransformerMTL().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

# train_model
def train_model(model, train_loader, val_loader, epochs=45):
    train_loss_history = []
    val_loss_history = []
    mod_acc_history = []
    sig_acc_history = []
    best_val_loss = float('inf')
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        mod_correct = 0
        sig_correct = 0
        total_samples = 0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} Training') as pbar:
            for X_batch, Y_mod_batch, Y_sig_batch in pbar:
                X_batch, Y_mod_batch, Y_sig_batch = X_batch.to(device), Y_mod_batch.to(device), Y_sig_batch.to(device)
                optimizer.zero_grad()

                if scaler and torch.cuda.is_available():
                    with torch.autocast(device_type='cuda'):
                        mod_output, sig_output = model(X_batch)
                        loss_mod = criterion(mod_output, Y_mod_batch)
                        loss_sig = criterion(sig_output, Y_sig_batch)
                        loss = 0.6 * loss_mod + 0.4 * loss_sig
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    mod_output, sig_output = model(X_batch)
                    loss_mod = criterion(mod_output, Y_mod_batch)
                    loss_sig = criterion(sig_output, Y_sig_batch)
                    loss = 0.6 * loss_mod + 0.4 * loss_sig
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                mod_pred = torch.argmax(mod_output, dim=1)
                sig_pred = torch.argmax(sig_output, dim=1)
                mod_correct += (mod_pred == Y_mod_batch).sum().item()
                sig_correct += (sig_pred == Y_sig_batch).sum().item()
                total_samples += Y_mod_batch.size(0)

                pbar.set_postfix({'Loss': loss.item()})

        train_loss = total_loss / len(train_loader)
        mod_acc = mod_correct / total_samples
        sig_acc = sig_correct / total_samples

        model.eval()
        val_loss = 0.0
        val_mod_correct = 0
        val_sig_correct = 0
        val_total = 0

        with tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} Validation') as pbar:
            with torch.no_grad():
                for X_val_batch, Y_mod_val_batch, Y_sig_val_batch in pbar:
                    X_val_batch, Y_mod_val_batch, Y_sig_val_batch = X_val_batch.to(device), Y_mod_val_batch.to(
                        device), Y_sig_val_batch.to(device)
                    mod_output, sig_output = model(X_val_batch)
                    val_loss_mod = criterion(mod_output, Y_mod_val_batch)
                    val_loss_sig = criterion(sig_output, Y_sig_val_batch)
                    val_loss += 0.6 * val_loss_mod + 0.4 * val_loss_sig

                    mod_pred = torch.argmax(mod_output, dim=1)
                    sig_pred = torch.argmax(sig_output, dim=1)
                    val_mod_correct += (mod_pred == Y_mod_val_batch).sum().item()
                    val_sig_correct += (sig_pred == Y_sig_val_batch).sum().item()
                    val_total += Y_mod_val_batch.size(0)

                    pbar.set_postfix({'Loss': val_loss.item()})

        val_loss = val_loss / len(val_loader)
        val_mod_acc = val_mod_correct / val_total
        val_sig_acc = val_sig_correct / val_total

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            print(f"New best model found at epoch {epoch + 1} with val loss: {val_loss:.4f}")
        else:
            print(f"No improvement at epoch {epoch + 1}. Keeping previous best model.")
            model.load_state_dict(best_model_weights)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        mod_acc_history.append((mod_acc, val_mod_acc))
        sig_acc_history.append((sig_acc, val_sig_acc))

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Mod Acc: {mod_acc:.4f} | Sig Acc: {sig_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Mod Acc: {val_mod_acc:.4f} | Val Sig Acc: {val_sig_acc:.4f}")

    return {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'mod_acc': mod_acc_history,
        'sig_acc': sig_acc_history
    }

history = train_model(model, train_loader, val_loader, epochs=45)

torch.save(model.state_dict(), 'weights/mtl_model_tracon.pth')

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot([acc[0] for acc in history['mod_acc']], label='Train Mod Acc')
plt.plot([acc[1] for acc in history['mod_acc']], label='Val Mod Acc')
plt.title('Modulation Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot([acc[0] for acc in history['sig_acc']], label='Train Sig Acc')
plt.plot([acc[1] for acc in history['sig_acc']], label='Val Sig Acc')
plt.title('Signal Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
train_loss = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in history['train_loss']]
val_loss = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in history['val_loss']]
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss Curve')
plt.legend()

plt.tight_layout()
plt.savefig('figure/accuracy_loss_curves_con.png')
plt.show()

