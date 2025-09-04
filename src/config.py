from pathlib import Path
import torch

# Path setup
base_dir = Path('/content/drive/My Drive/brain-images')
train_dir = base_dir / 'Processed' / 'Training'
test_dir = base_dir / 'Processed' / 'Testing'
model_save_path = Path('/content/model.pth')

