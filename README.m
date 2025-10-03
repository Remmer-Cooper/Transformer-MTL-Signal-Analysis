Project Structure

Core Scripts:
train_model.py: Main script for training the original CNN-based model (shared backbone).
validate_model.py: Script for validating the original CNN model.
train_transcon.py: Script for training the Vis_tran model.
val_transcon.py: Script for validating the Vis_tran model.

Supporting Directories:
data/: Contains data loading and preprocessing scripts.
datas/: Stores the actual datasets.
models/: Includes model definitions:
    Vis_tran model architecture
    Original CNN model architecture
    Model loading utilities for parameter configuration
utils/: Utility functions for:
    Confusion matrix visualization
    Training/validation helpers
    Other supporting functions

Additional Scripts:
    All other scripts are used for parameter tuning and debugging during development.




Getting Started

1. Environment Setup
Requirements:
    Python 3.9.20
    Recommended OS: Windows (includes Windows-specific components like pywin32)
    NVIDIA GPU with CUDA 12.4 support (for PyTorch GPU acceleration)

2. Dependency Installation
Use pip to install the required dependencies, including PyTorch, with the following commands:
    pip install torch==2.5.1+cu124
    pip install numpy==1.23.0
    pip install scikit-learn==1.5.2
    pip install h5py==3.12.1
    pip install matplotlib==3.9.2
    pip install seaborn==0.13.2
    pip install tqdm==4.67.1
    pip install einops==0.8.1
    pip install opencv-python==4.11.0.86
    pip install psutil==6.1.0

3. Running the Project
Execution Order: Always run training scripts before their corresponding validation scripts.

Original Model:
    bash
  python train_model.py       # Train the original model
  python validate_model.py    # Validate the original model

Vis_tran Model:
  bash
  python train_transcon.py    # Train the Vis_tran model
  python val_transcon.py      # Validate the Vis_tran model


