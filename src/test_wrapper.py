"""
This script performs the following tasks:
1. Loads preprocessed NumPy arrays (train/val)
2. Builds a sample hyperparameter dict
3. Calls evaluate_model and prints validation accuracy

Usage:
    python test_wrapper.py
"""

from train_wrapper import evaluate_model
import numpy as np

# Expect: train_X.npy, train_y.npy, val_X.npy, val_y.npy in CWD

# Load data
train_X = np.load("train_X.npy")
train_y = np.load("train_y.npy")
val_X = np.load("val_X.npy")
val_y = np.load("val_y.npy")

# Define a sample hyperparameter set (sanity check values; swap during optimization)
hparams = {
    "dense_units": 128,
    "dropout_rate": 0.4,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 3
}

# Run evaluation
val_acc = evaluate_model(hparams, train_X, train_y, val_X, val_y)

print(f"Validation Accuracy: {val_acc:.4f}")
