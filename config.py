from pathlib import Path
import torch

# Path setup
base_dir = Path('/content/drive/My Drive/brain-images')
train_dir = base_dir / 'Processed' / 'Training'
test_dir = base_dir / 'Processed' / 'Testing'
model_save_path = Path('/content/model.pth')

# Training parameters
batch_size = 32
epochs = 10
lr = 0.001
image_size = (224, 224)
num_classes = 4  # adjust if needed (this is for brain images)

# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')