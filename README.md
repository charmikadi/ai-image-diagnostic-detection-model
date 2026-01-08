# AI-Based Early Detection in Medical Imaging

A deep learning model for automated brain MRI image classification, designed to detect subtle, early-stage changes in medical imaging studies. This project uses a ResNet18-based architecture with transfer learning to identify early disease indicators and classify brain images into four categories: meningioma, pituitary, glioma, and no tumor.

## Overview

End-to-end machine learning pipeline for early-stage brain tumor detection and classification using PyTorch. The model assists clinicians in the diagnostic process by automatically categorizing brain MRI scans, emphasizing robustness, transparency, and alignment with clinical needs.

### Key Features

- **Multi-class Classification**: 4 categories (meningioma, pituitary, glioma, no tumor)
- **Early Detection**: Identifies subtle, early-stage changes not easily detected through conventional review
- **Transfer Learning**: Pre-trained ResNet18 with fine-tuning and hyperparameter optimization
- **Data Augmentation**: Rotation, flipping, and contrast adjustments for improved robustness
- **Comprehensive Metrics**: Precision, recall, F1-score, and support for thorough evaluation
- **GPU Support**: Automatic GPU detection and utilization
- **Colab Integration**: Jupyter notebook for easy experimentation

## Project Structure

```
ai-image-diagnostic-detection-model/
├── src/
│   ├── config.py          # Configuration settings
│   ├── dataset.py         # Dataset class and data loading
│   ├── model.py           # ResNet18 architecture
│   ├── train.py           # Training function
│   ├── evaluate.py        # Model evaluation
│   ├── main.py            # Main training script
│   └── utils.py           # Model I/O utilities
├── Trained_model_HSOC.ipynb  # Colab training notebook
└── trained_model_hsoc.py     # Trained model script
```

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (optional, recommended for training)

### Dependencies

```bash
pip install torch torchvision pillow matplotlib seaborn scikit-learn pandas numpy
```

## Dataset Structure

### Option 1: Directory Structure
```
brain-images/
├── Training/
│   ├── meningioma/
│   ├── pituitary/
│   ├── glioma/
│   └── notumor/
└── Testing/
    └── [same structure]
```

### Option 2: Processed with CSV
```
brain-images/
└── Processed/
    ├── Training/
    ├── Testing/
    └── image_labels.csv  # filename, label, split
```

## Configuration

Edit `src/config.py` to set paths and hyperparameters:

```python
base_dir = Path('/path/to/your/brain-images')
train_dir = base_dir / 'Training'
test_dir = base_dir / 'Testing'
model_save_path = Path('model.pth')

batch_size = 32
epochs = 10
lr = 0.001
image_size = (224, 224)
num_classes = 4
```

## Usage

### Local Training

1. Update paths in `src/config.py`
2. Run training:

```bash
cd src
python main.py
```

### Google Colab

1. Upload project to Google Colab
2. Mount Google Drive with dataset
3. Open `Trained_model_HSOC.ipynb`
4. Update paths and run all cells

### Evaluation

```python
from src.evaluate import evaluate_model
from src.utils import load_model
from src.model import get_resnet_model

model = get_resnet_model(num_classes=4)
model = load_model(model, 'model.pth', device='cuda')
accuracy, report = evaluate_model(model, test_loader, device)
```

## Model Architecture

- **Base**: Pre-trained ResNet18 (ImageNet weights)
- **Fine-tuning**: Freezes base layers, trains classifier only
- **Classifier**: FC(512→256) → ReLU → Dropout(0.4) → FC(256→4)

## Training

The training process includes:
- Preprocessing: Resizing and normalization
- Augmentation: Rotation, flipping, contrast adjustments
- Optimization: Adam optimizer (lr=0.001), Cross-entropy loss
- Iterative refinement through hyperparameter tuning
- Automatic model checkpointing

## Evaluation Metrics

- Overall accuracy
- Per-class precision, recall, F1-score, and support
- Classification report with detailed metrics

Critical for evaluating early-stage detection while minimizing false negatives.

## Model Performance

- Training samples: ~5,700 images
- Testing samples: ~1,300 images
- Classes: 4 (meningioma, pituitary, glioma, notumor)

## Model Saving and Loading

Models are saved automatically after training:

```python
from src.utils import load_model
from src.model import get_resnet_model

model = get_resnet_model(num_classes=4)
model = load_model(model, 'model.pth', device='cuda')
```

## Methodology

1. Literature review on AI in medical imaging
2. Dataset exploration and preprocessing (resizing, normalization)
3. Data augmentation (rotation, flipping, contrast adjustments)
4. Exploratory data analysis and label validation
5. Model selection and transfer learning
6. Training with hyperparameter tuning
7. Comprehensive evaluation using multiple metrics
8. Iterative optimization for improved generalization

## Development

### Code Structure

- **config.py**: Configuration management
- **dataset.py**: PyTorch Dataset class with preprocessing
- **model.py**: ResNet18 architecture with transfer learning
- **train.py**: Training loop with optimization
- **evaluate.py**: Metrics computation (precision, recall, F1-score, support)
- **utils.py**: Model I/O functions
- **main.py**: Training pipeline entry point

## License

Open source and available for research and educational purposes.

## Contributing

Contributions, issues, and feature requests are welcome!

## Contact

For questions or support, please open an issue on the repository.

---

**Note**: This model is intended for research and educational purposes. For clinical applications, ensure proper validation, regulatory compliance, and thorough testing. The model is designed to support clinicians, not replace clinical judgment.
