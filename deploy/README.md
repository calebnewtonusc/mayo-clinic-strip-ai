# Deployment Guide

This directory contains everything needed to deploy the Mayo Clinic STRIP AI model for production inference.

## Contents

- `api.py` - Flask REST API for model inference
- `api_client.py` - Python client for testing the API
- `Dockerfile` - Docker container configuration
- `docker-compose.yml` - Docker Compose orchestration
- `optimize_model.py` - Model optimization utilities (in scripts/)

## Quick Start

### Option 1: Run Locally

```bash
# Install additional dependencies
pip install flask flask-cors

# Run the API
python deploy/api.py --checkpoint checkpoints/best_model.pth
```

The API will be available at `http://localhost:5000`

### Option 2: Run with Docker

```bash
# Build and run
cd deploy
docker-compose up --build

# Or use Docker directly
docker build -t mayo-clinic-strip-ai -f Dockerfile ..
docker run -p 5000:5000 -v $(pwd)/models:/app/models mayo-clinic-strip-ai
```

## API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Model Information
```bash
curl http://localhost:5000/model_info
```

### Single Image Prediction
```bash
curl -X POST -F "file=@image.png" http://localhost:5000/predict
```

### Prediction with Uncertainty
```bash
curl -X POST -F "file=@image.png" -F "uncertainty=true" http://localhost:5000/predict
```

### Batch Prediction
```bash
curl -X POST -F "files=@image1.png" -F "files=@image2.png" http://localhost:5000/batch_predict
```

## Using the Python Client

```bash
# Health check
python deploy/api_client.py --action health

# Get model info
python deploy/api_client.py --action info

# Single prediction
python deploy/api_client.py --action predict --image data/test_image.png

# Prediction with uncertainty
python deploy/api_client.py --action predict --image data/test_image.png --uncertainty

# Batch prediction
python deploy/api_client.py --action batch_predict --images data/img1.png data/img2.png
```

## Model Optimization

Before deployment, optimize your model for faster inference:

```bash
# Quantization (reduces model size, faster CPU inference)
python scripts/optimize_model.py \
    --checkpoint checkpoints/best_model.pth \
    --method quantize \
    --output-dir models/optimized

# Pruning (removes unnecessary weights)
python scripts/optimize_model.py \
    --checkpoint checkpoints/best_model.pth \
    --method prune \
    --prune-amount 0.3 \
    --output-dir models/optimized

# Export to ONNX (cross-platform inference)
python scripts/optimize_model.py \
    --checkpoint checkpoints/best_model.pth \
    --export-onnx \
    --output-dir models/optimized
```

### Optimization Results

Typical optimization results:
- **Quantization**: 3-4x size reduction, 2-3x CPU speedup
- **Pruning (30%)**: 30% fewer parameters, modest speedup
- **ONNX**: Better compatibility, similar performance

## Production Deployment

### Environment Variables

```bash
FLASK_ENV=production
MODEL_CHECKPOINT=/path/to/model.pth
WORKERS=4
TIMEOUT=120
```

### Using Gunicorn (Recommended)

```bash
gunicorn --bind 0.0.0.0:5000 \
         --workers 4 \
         --timeout 120 \
         --access-logfile - \
         deploy.api:app
```

### NGINX Configuration (Optional)

For production, place NGINX in front of the API:

```nginx
upstream api {
    server localhost:5000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

## Security Considerations

### HIPAA Compliance

When deploying for medical use:

1. **Data Encryption**
   - Use HTTPS/TLS for all communications
   - Encrypt data at rest
   - Use encrypted volumes for Docker

2. **Access Control**
   - Implement authentication (JWT tokens, API keys)
   - Use role-based access control
   - Audit all access logs

3. **Data Privacy**
   - Never log patient identifiable information
   - Implement data retention policies
   - Use secure deletion methods

4. **Network Security**
   - Use VPN or private networks
   - Implement firewalls
   - Regular security audits

### Example: Adding Authentication

```python
# Add to api.py
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... existing code
```

## Monitoring

### Logging

The API logs all requests. Configure logging:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Metrics to Track

- Request latency
- Throughput (requests/second)
- Error rate
- Model confidence distribution
- Prediction class distribution

### Health Checks

The `/health` endpoint provides:
- API status
- Model loaded status
- (Optional) GPU availability
- (Optional) Memory usage

## Scaling

### Horizontal Scaling

Use Docker Compose to run multiple instances:

```yaml
services:
  api:
    deploy:
      replicas: 3
    # ... rest of config
```

### Load Balancing

Add NGINX or HAProxy for load balancing:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    depends_on:
      - api
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use model quantization
   - Add more RAM or use swap

2. **Slow Inference**
   - Use GPU if available
   - Apply model optimization
   - Reduce image size
   - Use batching

3. **Model Not Loading**
   - Check checkpoint path
   - Verify model architecture matches
   - Check file permissions

### Debug Mode

Run with debug logging:

```bash
python deploy/api.py \
    --checkpoint checkpoints/best_model.pth \
    --debug
```

## Cloud Deployment

### AWS

```bash
# Push to ECR
aws ecr create-repository --repository-name mayo-clinic-strip-ai
docker tag mayo-clinic-strip-ai:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/mayo-clinic-strip-ai:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/mayo-clinic-strip-ai:latest

# Deploy to ECS or use AWS Lambda
```

### Google Cloud

```bash
# Push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/mayo-clinic-strip-ai
gcloud run deploy --image gcr.io/PROJECT-ID/mayo-clinic-strip-ai --platform managed
```

### Azure

```bash
# Push to ACR
az acr build --registry <registry-name> --image mayo-clinic-strip-ai .
az container create --resource-group myResourceGroup --name mayo-clinic-strip-ai --image <registry-name>.azurecr.io/mayo-clinic-strip-ai
```

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | Inference Time | Throughput |
|----------|---------------|------------|
| CPU (Intel i7) | 50-100ms | ~10-20 req/s |
| GPU (NVIDIA RTX 3090) | 5-10ms | ~100-200 req/s |
| Quantized CPU | 20-40ms | ~25-50 req/s |

## Next Steps

1. Set up monitoring and logging
2. Implement authentication
3. Add rate limiting
4. Set up CI/CD pipeline
5. Configure automatic scaling
6. Set up backup and disaster recovery
