#!/bin/bash

# Production deployment script for Mayo Clinic STRIP AI
# Automates the deployment of the ML API with monitoring stack

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${GREEN}[DEPLOY]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_TYPE="${1:-docker}"  # docker, kubernetes, or local
ENVIRONMENT="${2:-production}"  # production, staging, or development

# Detect Python command
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null || echo "python3")
PIP=$(which pip3 2>/dev/null || which pip 2>/dev/null || echo "pip3")

print_msg "Mayo Clinic STRIP AI - Deployment Script"
print_msg "========================================"
print_msg "Deployment type: $DEPLOYMENT_TYPE"
print_msg "Environment: $ENVIRONMENT"
print_msg "========================================"

# Check prerequisites
check_prerequisites() {
    print_msg "Checking prerequisites..."

    # Check Docker
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        if ! command -v docker &> /dev/null; then
            print_error "Docker is not installed. Please install Docker first."
            exit 1
        fi
        if ! command -v docker-compose &> /dev/null; then
            print_error "Docker Compose is not installed. Please install Docker Compose first."
            exit 1
        fi
        print_msg "✓ Docker and Docker Compose are installed"
    fi

    # Check Python
    if [ "$DEPLOYMENT_TYPE" == "local" ]; then
        if ! command -v python3 &> /dev/null; then
            print_error "Python 3 is not installed."
            exit 1
        fi
        print_msg "✓ Python 3 is installed"
    fi

    # Check model checkpoint exists
    if [ ! -f "$PROJECT_ROOT/experiments/best_model.pth" ] && [ ! -f "$SCRIPT_DIR/models/best_model.pth" ]; then
        print_warning "Model checkpoint not found. You'll need to train a model first or place it in deploy/models/"
    fi
}

# Docker deployment
deploy_docker() {
    print_msg "Starting Docker deployment..."

    cd "$SCRIPT_DIR"

    # Check if .env file exists, create if not
    if [ ! -f .env ]; then
        print_msg "Creating .env file..."
        cat > .env <<EOF
# API Configuration
API_KEY=${API_KEY:-}
FLASK_ENV=${ENVIRONMENT}

# Grafana Configuration
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-admin}

# Model Configuration
MODEL_CHECKPOINT=/app/models/best_model.pth
EOF
        print_msg "✓ Created .env file"
    fi

    # Create necessary directories
    print_msg "Creating directories..."
    mkdir -p models logs prometheus-data grafana-data

    # Copy model if it exists in experiments
    if [ -f "$PROJECT_ROOT/experiments/best_model.pth" ] && [ ! -f "$SCRIPT_DIR/models/best_model.pth" ]; then
        print_msg "Copying model checkpoint..."
        cp "$PROJECT_ROOT/experiments/best_model.pth" "$SCRIPT_DIR/models/"
        print_msg "✓ Model copied"
    fi

    # Build and start services
    print_msg "Building Docker images..."
    docker-compose -f docker-compose-full.yml build

    print_msg "Starting services..."
    docker-compose -f docker-compose-full.yml up -d

    print_msg "Waiting for services to be healthy..."
    sleep 10

    # Check service health
    check_service_health
}

# Local deployment
deploy_local() {
    print_msg "Starting local deployment..."

    cd "$PROJECT_ROOT"

    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_msg "Creating virtual environment..."
        $PYTHON -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install dependencies
    print_msg "Installing dependencies..."
    $PIP install -r requirements.txt

    # Set environment variables
    export FLASK_ENV=${ENVIRONMENT}
    export API_KEY=${API_KEY:-}
    export MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-experiments/best_model.pth}

    # Start API server
    print_msg "Starting API server..."
    $PYTHON deploy/api_with_metrics.py \
        --checkpoint "$MODEL_CHECKPOINT" \
        --host 0.0.0.0 \
        --port 5000 \
        --log-dir logs &

    API_PID=$!
    print_msg "API server started (PID: $API_PID)"

    # Save PID for shutdown
    echo $API_PID > /tmp/mayo_api.pid

    print_msg "Waiting for API to be ready..."
    sleep 5

    check_service_health
}

# Check service health
check_service_health() {
    print_msg "Checking service health..."

    # Check API
    if curl -s -f http://localhost:5000/health > /dev/null; then
        print_msg "✓ API is healthy"
    else
        print_error "API health check failed"
        return 1
    fi

    # Check Prometheus (if Docker deployment)
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        if curl -s -f http://localhost:9090/-/healthy > /dev/null; then
            print_msg "✓ Prometheus is healthy"
        else
            print_warning "Prometheus health check failed"
        fi

        # Check Grafana
        if curl -s -f http://localhost:3000/api/health > /dev/null; then
            print_msg "✓ Grafana is healthy"
        else
            print_warning "Grafana health check failed"
        fi
    fi

    print_msg "Service health check complete"
}

# Show deployment info
show_deployment_info() {
    print_msg ""
    print_msg "========================================"
    print_msg "Deployment Complete!"
    print_msg "========================================"
    print_msg "API Endpoints:"
    print_msg "  - Health:      http://localhost:5000/health"
    print_msg "  - Predict:     http://localhost:5000/predict"
    print_msg "  - Metrics:     http://localhost:5000/metrics"
    print_msg "  - Model Info:  http://localhost:5000/model_info"

    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        print_msg ""
        print_msg "Monitoring:"
        print_msg "  - Prometheus:  http://localhost:9090"
        print_msg "  - Grafana:     http://localhost:3000"
        print_msg "                 (username: admin, password: ${GRAFANA_PASSWORD:-admin})"
    fi

    print_msg ""
    print_msg "Test the API:"
    print_msg "  curl http://localhost:5000/health"
    print_msg ""
    print_msg "To stop the deployment:"
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        print_msg "  cd $SCRIPT_DIR && docker-compose -f docker-compose-full.yml down"
    else
        print_msg "  $SCRIPT_DIR/shutdown.sh"
    fi
    print_msg "========================================"
}

# Main deployment flow
main() {
    check_prerequisites

    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        local)
            deploy_local
            ;;
        *)
            print_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            print_msg "Usage: $0 [docker|local] [production|staging|development]"
            exit 1
            ;;
    esac

    show_deployment_info
}

# Run main function
main
