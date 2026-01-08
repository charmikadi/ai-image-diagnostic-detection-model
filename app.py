"""
Streamlit Web Application for Brain MRI Tumor Classification
Interactive demo with GradCAM visualization
"""
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from model import get_resnet_model
from utils import load_model
from gradcam import apply_gradcam_to_image, GradCAM
from config import image_size, num_classes

# Page configuration
st.set_page_config(
    page_title="Brain MRI Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class names (update based on your dataset)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Default model path
DEFAULT_MODEL_PATH = "model.pth"

@st.cache_resource
def load_model_cached(model_path: str, device: torch.device):
    """Load and cache the model."""
    try:
        model = get_resnet_model(num_classes=num_classes).to(device)
        model = load_model(model, model_path, device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure you have a trained model saved as 'model.pth' in the project root.")
        return None

def get_transform():
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def get_target_layer(model):
    """Find the last convolutional layer for GradCAM."""
    # For ResNet18, use layer4 (last residual block)
    if hasattr(model, 'layer4'):
        target_layer = model.layer4[-1].conv2  # Last conv layer in last block
    else:
        # Fallback: find last conv layer
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        if target_layer is None:
            raise ValueError("Could not find convolutional layer in model")
    return target_layer

def main():
    # Title and description
    st.title("🧠 Brain MRI Tumor Classification")
    st.markdown("""
    Upload a brain MRI image to classify tumor types using deep learning.
    The model can identify: **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**.
    
    This demo includes GradCAM visualization showing which regions of the image 
    the model focuses on for its prediction.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_path = st.text_input("Model Path", value=DEFAULT_MODEL_PATH)
        
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a ResNet18-based deep learning model 
        trained on brain MRI scans for tumor classification.
        
        **Features:**
        - Multi-class classification (4 tumor types)
        - GradCAM interpretability visualization
        - Confidence scores and class probabilities
        """)
        
        st.header("📚 Instructions")
        st.markdown("""
        1. Upload a brain MRI image (JPG/PNG)
        2. Click 'Classify' to get predictions
        3. View GradCAM heatmap showing model attention
        4. Check class probabilities
        """)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        st.sidebar.success(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.info("ℹ️ Using CPU (GPU recommended for faster inference)")
    
    # Load model
    model = load_model_cached(model_path, device)
    
    if model is None:
        st.warning("⚠️ Model not loaded. Please train a model first or provide a valid model path.")
        st.stop()
    
    # File uploader
    st.header("📤 Upload Brain MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan image"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, caption="Input Brain MRI Scan", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Prediction Results")
            
            if st.button("🔬 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        # Preprocess image
                        transform = get_transform()
                        input_tensor = transform(image).unsqueeze(0).to(device)
                        
                        # Get prediction
                        with torch.no_grad():
                            output = model(input_tensor)
                            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
                            predicted_idx = output.argmax(dim=1).item()
                            confidence = probabilities[predicted_idx]
                        
                        # Display prediction
                        predicted_class = CLASS_NAMES[predicted_idx]
                        
                        # Color code based on prediction
                        if predicted_class == 'notumor':
                            color = "🟢"
                            status = "No Tumor Detected"
                        else:
                            color = "🔴"
                            status = f"Tumor Detected: {predicted_class.title()}"
                        
                        st.markdown(f"### {color} {status}")
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        
                        # Class probabilities
                        st.markdown("#### Class Probabilities")
                        prob_dict = {name: prob for name, prob in zip(CLASS_NAMES, probabilities)}
                        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        for class_name, prob in sorted_probs:
                            prob_bar = prob
                            label = f"**{class_name.title()}:** {prob:.2%}"
                            if class_name == predicted_class:
                                label = f"👉 {label}"
                            st.progress(prob, text=label)
                        
                        # Store results in session state for GradCAM
                        st.session_state['predicted_idx'] = predicted_idx
                        st.session_state['probabilities'] = probabilities
                        st.session_state['confidence'] = confidence
                        st.session_state['image'] = image
                        st.session_state['predicted_class'] = predicted_class
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        
        # GradCAM Visualization
        if 'predicted_idx' in st.session_state:
            st.header("🎯 Model Interpretability (GradCAM)")
            st.markdown("""
            The heatmap below shows which regions of the image the model focuses on 
            when making its prediction. **Red/yellow** areas indicate high attention, 
            while **blue** areas indicate low attention.
            """)
            
            try:
                transform = get_transform()
                target_layer = get_target_layer(model)
                
                with st.spinner("Generating GradCAM visualization..."):
                    overlay, _, _ = apply_gradcam_to_image(
                        model,
                        st.session_state['image'],
                        transform,
                        device,
                        target_layer=target_layer,
                        class_idx=st.session_state['predicted_idx'],
                        alpha=0.4
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🔍 Original Image")
                        st.image(st.session_state['image'], use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🎨 GradCAM Heatmap Overlay")
                        st.image(overlay, use_container_width=True)
                    
                    st.info(f"""
                    **Prediction:** {st.session_state['predicted_class'].title()}  
                    **Confidence:** {st.session_state['confidence']:.2%}  
                    **Explanation:** The model focuses on the highlighted regions to make this classification.
                    """)
                    
            except Exception as e:
                st.error(f"Error generating GradCAM: {str(e)}")
                st.info("GradCAM visualization requires a trained model with convolutional layers.")
    
    else:
        # Show example when no image is uploaded
        st.info("👆 Please upload a brain MRI image to get started.")
        
        # Example section
        with st.expander("ℹ️ What types of images work best?"):
            st.markdown("""
            - **Format:** JPG, JPEG, or PNG
            - **Content:** Brain MRI scans (axial, coronal, or sagittal views)
            - **Quality:** Clear, well-contrasted images work best
            - **Size:** Images will be resized to 224x224 pixels
            """)

if __name__ == "__main__":
    main()

