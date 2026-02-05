"""Production Flask API with Prometheus metrics and monitoring.

This enhanced API includes:
- Prometheus metrics for monitoring
- Request/response time tracking
- Error rate monitoring
- Model performance metrics
- Health checks
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import sys
import os
import logging
import signal
import atexit
import time
from datetime import datetime
from pathlib import Path
from functools import wraps
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import SimpleCNN, ResNetClassifier, EfficientNetClassifier
from src.evaluation.uncertainty import monte_carlo_dropout


# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://frontend-aus71qall-calebs-projects-a6310ab2.vercel.app",
            "https://frontend-mauve-seven-92.vercel.app",
            "http://localhost:3000"
        ]
    }
})

# Security: Set maximum file upload size (16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Prometheus metrics
REQUESTS_TOTAL = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method']
)

PREDICTIONS_TOTAL = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model', 'predicted_class']
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Prediction confidence scores',
    ['model', 'predicted_class'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

ACTIVE_REQUESTS = Gauge(
    'api_active_requests',
    'Number of active requests'
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether model is loaded (1) or not (0)'
)

BATCH_SIZE = Histogram(
    'batch_prediction_size',
    'Size of batch predictions',
    buckets=[1, 5, 10, 20, 50, 100]
)

ERROR_TOTAL = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['endpoint', 'error_type']
)

# Global model variable
model = None
device = None
model_config = None

# API authentication
API_KEY = os.environ.get('API_KEY', None)


def track_metrics(endpoint_name: str):
    """Decorator to track request metrics."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            ACTIVE_REQUESTS.inc()
            start_time = time.time()

            try:
                response = f(*args, **kwargs)

                # Track successful request
                duration = time.time() - start_time
                REQUEST_DURATION.labels(
                    endpoint=endpoint_name,
                    method=request.method
                ).observe(duration)

                # Determine status code
                if isinstance(response, tuple):
                    status = response[1]
                else:
                    status = 200

                REQUESTS_TOTAL.labels(
                    endpoint=endpoint_name,
                    method=request.method,
                    status=status
                ).inc()

                return response

            except Exception as e:
                # Track error
                ERROR_TOTAL.labels(
                    endpoint=endpoint_name,
                    error_type=type(e).__name__
                ).inc()

                REQUESTS_TOTAL.labels(
                    endpoint=endpoint_name,
                    method=request.method,
                    status=500
                ).inc()

                raise

            finally:
                ACTIVE_REQUESTS.dec()

        return decorated_function
    return decorator


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if API_KEY is None:
            return f(*args, **kwargs)

        provided_key = request.headers.get('X-API-Key')
        if not provided_key:
            REQUESTS_TOTAL.labels(
                endpoint=request.endpoint,
                method=request.method,
                status=401
            ).inc()
            return jsonify({'error': 'API key required. Provide X-API-Key header.'}), 401

        if provided_key != API_KEY:
            REQUESTS_TOTAL.labels(
                endpoint=request.endpoint,
                method=request.method,
                status=403
            ).inc()
            return jsonify({'error': 'Invalid API key'}), 403

        return f(*args, **kwargs)
    return decorated_function


def load_model(checkpoint_path):
    """Load model from checkpoint."""
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

    # Update model loaded metric
    MODEL_LOADED.set(1)

    config = {
        'arch': model_arch,
        'num_classes': num_classes,
        'class_names': ['Cardioembolic (CE)', 'Large Artery Atherosclerosis (LAA)']
    }

    return model, config


def preprocess_image(image_bytes):
    """Preprocess image for inference."""
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_np = np.array(image)

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
    image_tensor = transformed['image'].unsqueeze(0)

    return image_tensor


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/health', methods=['GET'])
@track_metrics('health')
def health_check():
    """Health check endpoint with detailed status."""
    health_status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized',
        'timestamp': datetime.now().isoformat()
    }

    if model is not None:
        health_status['model_info'] = {
            'architecture': model_config.get('arch'),
            'num_classes': model_config.get('num_classes')
        }

    return jsonify(health_status)


@app.route('/predict', methods=['POST'])
@track_metrics('predict')
@require_api_key
def predict():
    """Prediction endpoint with metrics tracking."""
    if model is None or model_config is None or device is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    compute_uncertainty = request.form.get('uncertainty', 'false').lower() == 'true'

    try:
        # Read and preprocess image
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)

        # Inference
        if compute_uncertainty:
            mean_pred, std_pred, _ = monte_carlo_dropout(
                model, image_tensor, n_iterations=30
            )

            probs = mean_pred[0]
            uncertainty = std_pred[0]

            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            # Track metrics
            PREDICTIONS_TOTAL.labels(
                model=model_config['arch'],
                predicted_class=model_config['class_names'][predicted_class]
            ).inc()

            PREDICTION_CONFIDENCE.labels(
                model=model_config['arch'],
                predicted_class=model_config['class_names'][predicted_class]
            ).observe(confidence)

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

            # Track metrics
            PREDICTIONS_TOTAL.labels(
                model=model_config['arch'],
                predicted_class=model_config['class_names'][predicted_class]
            ).inc()

            PREDICTION_CONFIDENCE.labels(
                model=model_config['arch'],
                predicted_class=model_config['class_names'][predicted_class]
            ).observe(confidence)

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
        ERROR_TOTAL.labels(
            endpoint='predict',
            error_type=type(e).__name__
        ).inc()
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
@track_metrics('batch_predict')
@require_api_key
def batch_predict():
    """Batch prediction endpoint with metrics."""
    if model is None or model_config is None or device is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'Empty file list'}), 400

    # Track batch size
    BATCH_SIZE.observe(len(files))

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

            # Track metrics
            PREDICTIONS_TOTAL.labels(
                model=model_config['arch'],
                predicted_class=model_config['class_names'][predicted_class]
            ).inc()

            PREDICTION_CONFIDENCE.labels(
                model=model_config['arch'],
                predicted_class=model_config['class_names'][predicted_class]
            ).observe(confidence)

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
        ERROR_TOTAL.labels(
            endpoint='batch_predict',
            error_type=type(e).__name__
        ).inc()
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
@track_metrics('model_info')
def model_info():
    """Get model information."""
    if model is None or model_config is None or device is None:
        return jsonify({'error': 'Model not loaded'}), 500

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
    """Main function to start the API with metrics."""
    import argparse

    parser = argparse.ArgumentParser(description='Run inference API with Prometheus metrics')
    parser.add_argument('--checkpoint', type=str,
                        default=os.environ.get('MODEL_CHECKPOINT'),
                        help='Path to model checkpoint')
    parser.add_argument('--host', type=str,
                        default=os.environ.get('API_HOST', '0.0.0.0'),
                        help='Host to run on')
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('API_PORT', '5000')),
                        help='Port to run on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--log-dir', type=str,
                        default=os.environ.get('LOG_DIR', 'logs'),
                        help='Directory for log files')
    args = parser.parse_args()

    if args.checkpoint is None:
        parser.error('--checkpoint is required (or set MODEL_CHECKPOINT environment variable)')

    # Setup logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'api_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Load model (optional - API can start without model)
    logger.info(f"Attempting to load model from {args.checkpoint}...")
    global model, model_config
    try:
        model, model_config = load_model(args.checkpoint)
        logger.info(f"Model loaded successfully: {model_config['arch']}")
        logger.info(f"Device: {device}")
    except FileNotFoundError:
        logger.warning(f"Model file not found: {args.checkpoint}")
        logger.warning("API will start without a model - predictions will fail until model is uploaded")
        model = None
        model_config = None
    logger.info(f"Logging to: {log_file}")

    # Setup graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info(f"Received shutdown signal {signum}")
        logger.info("Cleaning up resources...")
        global model
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        MODEL_LOADED.set(0)
        logger.info("Shutdown complete")
        sys.exit(0)

    def cleanup():
        logger.info("API shutdown - cleanup complete")

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    atexit.register(cleanup)

    # Start API
    logger.info(f"Starting API on {args.host}:{args.port}")
    logger.info("Available endpoints:")
    logger.info("  GET  /health           - Health check")
    logger.info("  GET  /model_info       - Model information")
    logger.info("  GET  /metrics          - Prometheus metrics")
    logger.info("  POST /predict          - Single image prediction")
    logger.info("  POST /batch_predict    - Batch image prediction")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
