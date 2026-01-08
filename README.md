# AI Image Diagnostic Detection Model

A deep learning model for automated brain MRI image classification, designed to identify and classify brain tumor types from medical imaging data. This project uses a ResNet18-based architecture with transfer learning to classify brain images into four categories: meningioma, pituitary, glioma, and no tumor.

## Overview

This project implements an end-to-end machine learning pipeline for medical image analysis, specifically focused on brain tumor detection and classification. The model leverages PyTorch and deep learning techniques to assist in the diagnostic process by automatically categorizing brain MRI scans.

### Key Features

- **Multi-class Classification**: Classifies brain images into 4 categories:
  - Meningioma
  - Pituitary
  - Glioma
  - No Tumor

- **Transfer Learning**: Utilizes pre-trained ResNet18 architecture with fine-tuning
- **GPU Support**: Automatic GPU detection and utilization when available
- **Comprehensive Pipeline**: Complete training, evaluation, and model saving functionality
- **Colab Integration**: Jupyter notebook included for easy experimentation in Google Colab
- **Flexible Dataset Handling**: Supports both organized directory structure and CSV-based datasets

## Project Structure

```
ai-image-diagnostic-detection-model/
├── src/
│   ├── config.py          # Configuration settings (paths, hyperparameters)
│   ├── dataset.py         # Dataset class and data loading utilities
│   ├── model.py           # ResNet18 model architecture
│   ├── train.py           # Training function
│   ├── evaluate.py        # Model evaluation and metrics
│   ├── main.py            # Main training script
│   └── utils.py           # Utility functions (save/load model)
├── brain/
│   └── brain-images.html  # Reference to dataset location
├── Trained_model_HSOC.ipynb  # Jupyter notebook for Colab training
├── trained_model_hsoc.py     # Trained model script
├── ai-imaging-diagnosis.zip  # Project archive
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (optional, but recommended for training)

### Dependencies

Install the required packages:

```bash
pip install torch torchvision tensorflow keras opencv-python pillow matplotlib seaborn scikit-learn pandas numpy
```

Or install individually:

```bash
pip install torch torchvision pillow matplotlib seaborn scikit-learn pandas numpy
```

## Dataset Structure

The model expects data in one of the following formats:

### Option 1: Directory Structure
```
brain-images/
├── Training/
│   ├── meningioma/
│   ├── pituitary/
│   ├── glioma/
│   └── notumor/
└── Testing/
    ├── meningioma/
    ├── pituitary/
    ├── glioma/
    └── notumor/
```

### Option 2: Processed with CSV
```
brain-images/
└── Processed/
    ├── Training/
    ├── Testing/
    └── image_labels.csv  # Contains: filename, label, split
```

## Configuration

Edit `src/config.py` to set your paths and hyperparameters:

```python
base_dir = Path('/path/to/your/brain-images')
train_dir = base_dir / 'Training'
test_dir = base_dir / 'Testing'
model_save_path = Path('model.pth')

# Training parameters
batch_size = 32
epochs = 10
lr = 0.001
image_size = (224, 224)
num_classes = 4
```

## Usage

### Local Training

1. Update the paths in `src/config.py`
2. Run the training script:

```bash
cd src
python main.py
```

### Google Colab

1. Upload the project to Google Colab
2. Mount your Google Drive with the dataset
3. Open `Trained_model_HSOC.ipynb`
4. Update paths in the notebook cells
5. Run all cells

### Training with Custom Script

```python
from src.train import train

model, classes = train()
```

### Evaluation

```python
from src.evaluate import evaluate_model
from src.utils import load_model
from src.model import get_resnet_model

# Load model
model = get_resnet_model(num_classes=4)
model = load_model(model, 'model.pth', device='cuda')

# Evaluate
accuracy, report = evaluate_model(model, test_loader, device)
print(f"Accuracy: {accuracy}")
print(report)
```

## Model Architecture

The model is based on ResNet18 with the following modifications:

- **Base**: Pre-trained ResNet18 (ImageNet weights)
- **Fine-tuning**: Freezes base layers initially, only trains classifier
- **Classifier**: 
  - Fully Connected Layer (512 → 256)
  - ReLU Activation
  - Dropout (0.4)
  - Final Classification Layer (256 → 4 classes)

## Training

The training process includes:
- Data augmentation (resize, normalization)
- Batch processing with configurable batch size
- Adam optimizer with learning rate 0.001
- Cross-entropy loss function
- Model checkpointing after training

## Evaluation Metrics

The model provides:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Classification report with detailed metrics

## Model Performance

Training on the brain MRI dataset:
- Training samples: ~5,700 images
- Testing samples: ~1,300 images
- Classes: 4 (meningioma, pituitary, glioma, notumor)

## Model Saving and Loading

Models are saved automatically after training. To load a saved model:

```python
from src.utils import load_model
from src.model import get_resnet_model

model = get_resnet_model(num_classes=4)
model = load_model(model, 'model.pth', device='cuda')
```

## Development

### Code Structure

- **config.py**: Centralized configuration management
- **dataset.py**: Custom PyTorch Dataset class for MRI images
- **model.py**: Model architecture definition
- **train.py**: Training loop implementation
- **evaluate.py**: Evaluation and metrics computation
- **utils.py**: Helper functions for model I/O
- **main.py**: Entry point for training pipeline

## 📄 License

This project is open source and available for research and educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## Contact

For questions or support, please open an issue on the repository.

---

**Note**: This model is intended for research and educational purposes. For clinical applications, please ensure proper validation and regulatory compliance.
