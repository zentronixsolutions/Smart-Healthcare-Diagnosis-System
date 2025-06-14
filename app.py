from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from timm import create_model
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger().setLevel(logging.ERROR)

app = Flask(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained models with weights_only=True to address security warnings
try:
    mri_model = create_model("swinv2_tiny_window16_256", pretrained=False, num_classes=10)
    mri_model.load_state_dict(torch.load("models/mri35.pth", map_location=device, weights_only=True))
    mri_model.to(device)
    mri_model.eval()

    xray_model = create_model("swinv2_tiny_window16_256", pretrained=False, num_classes=4)
    xray_model.load_state_dict(torch.load("models/chest_xray.pth", map_location=device, weights_only=True))
    xray_model.to(device)
    xray_model.eval()
except Exception as e:
    print(f"Error loading models: {e}")

# Define labels for each model
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
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_prediction(image, model_type):
    """Process image and get prediction"""
    # Select model and labels
    if model_type == "mri":
        model = mri_model
        labels = mri_labels
    else:  # xray
        model = xray_model
        labels = xray_labels
    
    # Process image
    img = Image.open(image).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()
        predicted_class = int(np.argmax(probabilities))
    
    # Create confidence data
    confidence_data = {labels[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    # Generate confidence chart with better styling
    plt.figure(figsize=(10, 6), facecolor='#f8f9fa')
    
    # Sort classes by probability for better visualization
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probs = probabilities[sorted_indices]
    
    bars = plt.barh(range(len(sorted_probs)), sorted_probs, color='#4361ee', alpha=0.7)
    plt.yticks(range(len(sorted_probs)), sorted_labels, fontsize=10)
    plt.xlabel("Confidence Score", fontsize=12)
    plt.title("Condition Probability Distribution", fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add percentage labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getbuffer()).decode('utf-8')
    plt.close()
    
    # Convert image to base64 string
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        'predicted_class': labels[predicted_class],
        'confidence_data': confidence_data,
        'plot_base64': plot_base64,
        'image_base64': img_base64
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'mri')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        result = get_prediction(file, model_type)
        return jsonify(result)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Ensure the templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created templates directory")
    
    # Print the current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Templates directory location: {os.path.join(os.getcwd(), 'templates')}")
    
    app.run(debug=True)