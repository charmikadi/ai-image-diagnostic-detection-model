import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import batch_size, image_size, base_dir

class MRIDataset(Dataset):
    def __init__(self, csv_file, split, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split]  # filter for 'training' or 'testing'
        self.transform = transform

        # Create label to index mapping
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}
        self.df['label_idx'] = self.df['label'].map(self.label2idx)

        self.split = split
        self.image_dir = os.path.join(base_dir, 'Processed', split.capitalize())  # e.g. Processed/Training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = row['label_idx']

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(csv_file=os.path.join(base_dir, 'Processed', 'image_labels.csv')):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # for RGB
    ])

    train_dataset = MRIDataset(csv_file=csv_file, split='training', transform=transform)
    test_dataset = MRIDataset(csv_file=csv_file, split='testing', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.label2idx
