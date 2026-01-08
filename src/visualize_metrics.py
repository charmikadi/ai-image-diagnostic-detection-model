import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_curve, 
    auc, 
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
import seaborn as sns


def evaluate_model_with_roc(model, data_loader, device, class_names, save_path=None):
    """
    Evaluate model and generate ROC curves for multi-class classification.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: List of class names
        save_path: Optional path to save ROC curve figure
    
    Returns:
        dict: Dictionary containing accuracy, classification report, and ROC data
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # For multi-class ROC, use One-vs-Rest (OvR) approach
    n_classes = len(class_names)
    
    # Binarize labels for one-vs-rest ROC
    y_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-averaged ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-averaged ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i in range(n_classes):
        plt.plot(
            fpr[i], 
            tpr[i], 
            color=colors[i % len(colors)], 
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
        )
    
    # Plot micro and macro averaged ROC curves
    plt.plot(
        fpr["micro"], 
        tpr["micro"],
        color='deeppink', 
        linestyle='--', 
        lw=2,
        label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})'
    )
    plt.plot(
        fpr["macro"], 
        tpr["macro"],
        color='navy', 
        linestyle='--', 
        lw=2,
        label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})'
    )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    
    return {
        'accuracy': acc,
        'classification_report': report,
        'roc_auc': roc_auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix for multi-class classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save confusion matrix figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_class_metrics(y_true, y_pred, class_names):
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

