"""
GradCAM (Gradient-weighted Class Activation Mapping) visualization for model interpretability.
This module generates heatmaps showing which regions of the image the model focuses on.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visualizing model attention.
    """
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer to generate CAM (usually the last convolutional layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activations during forward pass."""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients during backward pass."""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None):
        """
        Generate CAM for the given input image.
        
        Args:
            input_image: Input tensor (1, C, H, W)
            class_idx: Class index to generate CAM for. If None, uses predicted class.
        
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Calculate weights (gradient-weighted)
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input image size
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        
        return cam, class_idx


def apply_gradcam_to_image(
    model, 
    image: Image.Image, 
    transform,
    device: torch.device,
    target_layer=None,
    class_idx: Optional[int] = None,
    alpha: float = 0.4
) -> Tuple[np.ndarray, int, float]:
    """
    Apply GradCAM to a PIL image and return visualization.
    
    Args:
        model: Trained PyTorch model
        image: PIL Image
        transform: Image preprocessing transform
        device: Device to run model on
        target_layer: Target layer for GradCAM (if None, automatically finds last conv layer)
        class_idx: Class index to visualize (if None, uses predicted class)
        alpha: Transparency for overlay
    
    Returns:
        Tuple of (overlay_image, predicted_class_idx, confidence_score)
    """
    model.eval()
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_idx = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_idx].item()
    
    # Find target layer if not provided (last convolutional layer in ResNet)
    if target_layer is None:
        target_layer = None
        # For ResNet18, we want layer4 (the last residual block)
        if hasattr(model, 'layer4'):
            target_layer = model.layer4[-1].conv2  # Last conv layer in last block
        else:
            # Fallback: find last conv layer
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
        if target_layer is None:
            raise ValueError("Could not find convolutional layer in model")
    
    # Generate CAM
    gradcam = GradCAM(model, target_layer)
    cam, visualized_class_idx = gradcam.generate_cam(input_tensor, class_idx)
    
    # Convert image to numpy array
    img_array = np.array(image.convert('RGB'))
    img_array = cv2.resize(img_array, (224, 224))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    
    return overlay, visualized_class_idx, confidence


def visualize_gradcam(
    model,
    image: Image.Image,
    transform,
    device: torch.device,
    class_names: list,
    target_layer=None,
    save_path: Optional[str] = None
):
    """
    Generate and display GradCAM visualization.
    
    Args:
        model: Trained PyTorch model
        image: PIL Image
        transform: Image preprocessing transform
        device: Device to run model on
        class_names: List of class names
        target_layer: Target layer for GradCAM
        save_path: Optional path to save the figure
    """
    overlay, pred_idx, confidence = apply_gradcam_to_image(
        model, image, transform, device, target_layer
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # GradCAM overlay
    axes[1].imshow(overlay)
    axes[1].set_title(
        f'GradCAM Overlay\nPredicted: {class_names[pred_idx]}\nConfidence: {confidence:.2%}',
        fontsize=12,
        fontweight='bold'
    )
    axes[1].axis('off')
    
    # Get class probabilities
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
    
    # Bar plot of probabilities
    axes[2].barh(class_names, probabilities)
    axes[2].set_xlabel('Probability', fontsize=10)
    axes[2].set_title('Class Probabilities', fontsize=12, fontweight='bold')
    axes[2].set_xlim(0, 1)
    for i, prob in enumerate(probabilities):
        axes[2].text(prob + 0.01, i, f'{prob:.2%}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GradCAM visualization saved to {save_path}")
    
    plt.show()
    
    return overlay, pred_idx, confidence, probabilities

