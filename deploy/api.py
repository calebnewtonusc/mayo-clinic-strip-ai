"""Flask API for stroke classification model inference."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import SimpleCNN, ResNetClassifier, EfficientNetClassifier
from src.evaluation.uncertainty import monte_carlo_dropout


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model variable
model = None
device = None
model_config = None


def load_model(checkpoint_path):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Loaded model
    """
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model configuration
    model_arch = checkpoint.get('arch', 'resnet18')
    num_classes = checkpoint.get('num_classes', 2)

    # Initialize model
    if 'resnet' in model_arch:
        model = ResNetClassifier(
            arch=model_arch,
            num_classes=num_classes,
            pretrained=False
        )
    elif 'efficientnet' in model_arch:
        model = EfficientNetClassifier(
            arch=model_arch,
            num_classes=num_classes,
            pretrained=False
        )
    else:
        model = SimpleCNN(in_channels=3, num_classes=num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    config = {
        'arch': model_arch,
        'num_classes': num_classes,
        'class_names': ['Cardioembolic (CE)', 'Large Artery Atherosclerosis (LAA)']
    }

    return model, config


def preprocess_image(image_bytes):
    """Preprocess image for inference.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy
    image_np = np.array(image)

    # Apply transforms
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    transformed = transform(image=image_np)
    image_tensor = transformed['image']

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint.

    Expects:
        - file: Image file
        - uncertainty: (optional) Whether to compute uncertainty

    Returns:
        JSON with predictions and probabilities
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Get options
    compute_uncertainty = request.form.get('uncertainty', 'false').lower() == 'true'

    try:
        # Read image
        image_bytes = file.read()

        # Preprocess
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)

        # Inference
        if compute_uncertainty:
            # Use MC dropout for uncertainty
            mean_pred, std_pred, _ = monte_carlo_dropout(
                model, image_tensor, n_iterations=30
            )

            probs = mean_pred[0]
            uncertainty = std_pred[0]

            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            response = {
                'predicted_class': predicted_class,
                'class_name': model_config['class_names'][predicted_class],
                'confidence': confidence,
                'probabilities': {
                    model_config['class_names'][i]: float(probs[i])
                    for i in range(len(probs))
                },
                'uncertainty': {
                    model_config['class_names'][i]: float(uncertainty[i])
                    for i in range(len(uncertainty))
                }
            }
        else:
            # Standard inference
            with torch.no_grad():
                output = model(image_tensor)
                probs = F.softmax(output, dim=1)

            probs = probs.cpu().numpy()[0]
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            response = {
                'predicted_class': predicted_class,
                'class_name': model_config['class_names'][predicted_class],
                'confidence': confidence,
                'probabilities': {
                    model_config['class_names'][i]: float(probs[i])
                    for i in range(len(probs))
                }
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint.

    Expects:
        - files: Multiple image files

    Returns:
        JSON with predictions for all images
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Check if files are present
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'Empty file list'}), 400

    try:
        results = []

        for file in files:
            # Read and preprocess
            image_bytes = file.read()
            image_tensor = preprocess_image(image_bytes)
            image_tensor = image_tensor.to(device)

            # Inference
            with torch.no_grad():
                output = model(image_tensor)
                probs = F.softmax(output, dim=1)

            probs = probs.cpu().numpy()[0]
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            results.append({
                'filename': file.filename,
                'predicted_class': predicted_class,
                'class_name': model_config['class_names'][predicted_class],
                'confidence': confidence,
                'probabilities': {
                    model_config['class_names'][i]: float(probs[i])
                    for i in range(len(probs))
                }
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return jsonify({
        'architecture': model_config['arch'],
        'num_classes': model_config['num_classes'],
        'class_names': model_config['class_names'],
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': str(device)
    })


def main():
    """Main function to start the API."""
    import argparse

    parser = argparse.ArgumentParser(description='Run inference API')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    global model, model_config
    model, model_config = load_model(args.checkpoint)
    print(f"Model loaded: {model_config['arch']}")
    print(f"Device: {device}")

    # Start API
    print(f"\nStarting API on {args.host}:{args.port}")
    print("\nAvailable endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /model_info       - Model information")
    print("  POST /predict          - Single image prediction")
    print("  POST /batch_predict    - Batch image prediction")
    print("\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
