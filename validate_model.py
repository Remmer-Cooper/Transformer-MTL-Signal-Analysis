import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import h5py
import matplotlib.pyplot as plt
from models.build_mtl_model import MTLModel
from data.dataset_loader import load_data, preprocess_data, split_data
from utils.mltools1_1 import plot_confusion_matrix, calculate_confusion_matrix, plot_snr_confusion_matrix

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
X, Y_mod, Y_sig, mod_encoder, sig_encoder, SNRs = load_data(r'D:\Test\mywork\Muti_task_classify\datas\RadComDynamic\RadComDynamic.hdf5')
X_processed = preprocess_data(X)

# 划分数据集
(X_train, Y_mod_train, Y_sig_train, SNRs_train), \
(X_val, Y_mod_val, Y_sig_val, SNRs_val), \
(X_test, Y_mod_test, Y_sig_test, SNRs_test) = split_data(X_processed, Y_mod, Y_sig, SNRs)

# 转换为PyTorch张量并移动到设备
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_mod_test = torch.tensor(Y_mod_test, dtype=torch.long).to(device)  # 确保是 Long 类型
Y_sig_test = torch.tensor(Y_sig_test, dtype=torch.long).to(device)  # 确保是 Long 类型

# 初始化模型并加载权重
model = MTLModel().to(device)
# 设置 weights_only=True 避免安全警告
model.load_state_dict(torch.load('weights/mtl_model.pth', weights_only=True))
model.eval()

SNRs_test_np = SNRs_test

# 评估测试集
mod_preds = []
sig_preds = []
with torch.no_grad():
    for X_batch, _, _ in DataLoader(TensorDataset(X_test, Y_mod_test, Y_sig_test), batch_size=32):
        mod_output, sig_output = model(X_batch)
        mod_preds.extend(torch.argmax(mod_output, dim=1).cpu().numpy())
        sig_preds.extend(torch.argmax(sig_output, dim=1).cpu().numpy())

# 绘制混淆矩阵
plot_confusion_matrix(Y_mod_test.cpu().numpy(), mod_preds, 
                     title="Modulation Confusion Matrix",
                     classes=mod_encoder.classes_,
                     save_filename='figure/test_test_mod_confusion.png')

plot_confusion_matrix(Y_sig_test.cpu().numpy(), sig_preds,
                     title="Signal Confusion Matrix",
                     classes=sig_encoder.classes_,
                     save_filename='figure/test_test_sig_confusion.png')

# 在测试集评估后添加以下代码：

# 将测试数据移到CPU并转换为numpy
X_test_np = X_test.cpu().numpy()

# 绘制不同SNR下的调制分类混淆矩阵
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)  # 将输入数据移动到 GPU
plot_snr_confusion_matrix(
    model, 
    X_test_tensor,  # 传递 GPU 上的张量
    Y_mod_test.cpu().numpy(),
    SNRs_test_np,
    classes=mod_encoder.classes_,
    title_prefix="Modulation Confusion Matrix",
    save_dir='figure/modulation'
)

# 绘制不同SNR下的信号分类混淆矩阵
plot_snr_confusion_matrix(
    model,
    X_test_tensor,  # 传递 GPU 上的张量
    Y_sig_test.cpu().numpy(),
    SNRs_test_np,
    classes=sig_encoder.classes_,
    title_prefix="Signal Confusion Matrix",
    save_dir='figure/signal'
)