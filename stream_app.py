import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from timm import create_model
import logging
import warnings
import os

os.environ["STREAMLIT_SERVER_WATCH_CHANGES"] = "false"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Fix for PyTorch load warning by setting weights_only=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disable Streamlit's file watcher to prevent the torch.classes path error
# This must be done before importing any streamlit modules



# Load trained models with weights_only=True to address security warnings
mri_model = create_model("swinv2_tiny_window16_256", pretrained=False, num_classes=10)
mri_model.load_state_dict(torch.load("models/mri35.pth", map_location=device, weights_only=True))
mri_model.to(device)
mri_model.eval()

xray_model = create_model("swinv2_tiny_window16_256", pretrained=False, num_classes=4)
xray_model.load_state_dict(torch.load("models/chest_xray.pth", map_location=device, weights_only=True))
xray_model.to(device)
xray_model.eval()

mri_labels = [
    "Benign", "Glioma", "Malignancy", "Meningioma", "No Tumor",
    "Pituitary", "Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"
]

xray_labels = [
    "Bacterial Pneumonia", "Normal", "Tuberculosis", "Virus Pneumonia"
]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI setup
st.set_page_config(page_title="Smart Healthcare Diagnosis System", layout="wide")
st.title("Smart Healthcare Diagnosis System")
st.sidebar.header("Select Model")

# Initialize session state variables if they don't exist
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "MRI Model"
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None
if "probabilities" not in st.session_state:
    st.session_state.probabilities = None
if "model_changed" not in st.session_state:
    st.session_state.model_changed = False

# Function to handle model change
def change_model():
    st.session_state.model_changed = True
    st.session_state.uploaded_file = None
    st.session_state.predicted_class = None
    st.session_state.probabilities = None

# Sidebar model selection
model_choice = st.sidebar.radio(
    "Choose a model:",
    ("MRI Model", "Chest X-ray Model"),
    key="model_selector"
)

# Check if model has changed
if "previous_model" not in st.session_state:
    st.session_state.previous_model = model_choice
elif st.session_state.previous_model != model_choice:
    change_model()
    st.session_state.previous_model = model_choice
    # Force rerun to clear the UI
    st.rerun()

# Store the current model selection
st.session_state.selected_model = model_choice

# Set labels and model based on selection
labels = mri_labels if model_choice == "MRI Model" else xray_labels
model = mri_model if model_choice == "MRI Model" else xray_model

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key=f"uploader_{model_choice}")

# Update session state if a file is uploaded
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    # Reset model_changed flag
    st.session_state.model_changed = False

# Exit early if no file is uploaded
if st.session_state.uploaded_file is None:
    st.stop()

# Process the uploaded image
image = Image.open(st.session_state.uploaded_file).convert("RGB")

# Layout for image and prediction results
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    if model_choice == "MRI Model":
        st.markdown("<div style='display: flex; align-items: center; height: 100%; justify-content: center; margin-top:100px;'>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='display: flex; align-items: center; height: 100%; justify-content: center; margin-top:20px'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", width=300)
    st.markdown("</div>", unsafe_allow_html=True)

# Run the model prediction
input_tensor = transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()
    predicted_class = int(np.argmax(probabilities))

# Save predictions in session state
st.session_state.predicted_class = predicted_class
st.session_state.probabilities = probabilities

with col2:
    st.subheader("Prediction Results")
    st.markdown(
        f"**Predicted Class:** <span style='font-size: 1.5em; margin-left: 10px;'>`{labels[predicted_class]}`</span>",
        unsafe_allow_html=True,
    )
    st.subheader("Confidence Scores")
    confidence_data = {labels[i]: f"{prob * 100:.2f}%" for i, prob in enumerate(probabilities)}
    st.table(confidence_data)

st.subheader("Confidence Score Distribution")
fig, ax = plt.subplots(facecolor='#0E1117', figsize=(8, 4))
ax.barh(range(len(probabilities)), probabilities, color='#1f77b4')
ax.set_yticks(range(len(probabilities)))
ax.set_yticklabels(labels, color='white')
ax.set_xlabel("Confidence Score", color='white')
ax.set_ylabel("Class", color='white')
ax.tick_params(colors='white')
ax.set_xlim(0, 1)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
plt.style.use('dark_background')
st.pyplot(fig)