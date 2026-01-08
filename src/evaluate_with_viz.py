"""
Enhanced evaluation script with ROC curve visualization for multi-class classification.
Usage: python evaluate_with_viz.py
"""
import torch
from dataset import get_dataloaders
from model import get_resnet_model
from utils import load_model
from config import device, num_classes, model_save_path
from visualize_metrics import evaluate_model_with_roc, plot_confusion_matrix, plot_class_metrics


def main():
    # Load data
    train_loader, test_loader, label2idx = get_dataloaders()
    class_names = list(label2idx.keys())
    
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Device: {device}")
    
    # Load trained model
    print(f"\nLoading model from {model_save_path}...")
    model = get_resnet_model(num_classes=len(class_names)).to(device)
    model = load_model(model, model_save_path, device)
    
    print("\nEvaluating model on test set...")
    print("=" * 60)
    
    # Evaluate with ROC curves
    results = evaluate_model_with_roc(
        model, 
        test_loader, 
        device, 
        class_names,
        save_path='roc_curves.png'
    )
    
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    print("\nROC AUC Scores:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {results['roc_auc'][i]:.4f}")
    print(f"Micro-average: {results['roc_auc']['micro']:.4f}")
    print(f"Macro-average: {results['roc_auc']['macro']:.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        class_names,
        save_path='confusion_matrix.png'
    )
    
    # Plot per-class metrics
    print("\nGenerating per-class metrics...")
    plot_class_metrics(
        results['labels'],
        results['predictions'],
        class_names
    )
    
    print("\n" + "=" * 60)
    print("Evaluation complete! Check the generated plots.")


if __name__ == '__main__':
    main()

