import sys
import torch
import mne
import numpy as np

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("MNE version:", mne.__version__)
print("NumPy version:", np.__version__)

print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)

# Fake EEG batch
# Shape: [batch_size, channels, time_points]
x = torch.randn(4, 18, 1024).to(device)

print("Fake EEG batch shape:", x.shape)
print("Setup check complete.")