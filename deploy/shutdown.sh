#!/bin/bash

# Graceful shutdown script for Mayo Clinic STRIP AI deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_msg() {
    echo -e "${GREEN}[SHUTDOWN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_TYPE="${1:-auto}"  # auto, docker, or local

print_msg "Mayo Clinic STRIP AI - Shutdown Script"
print_msg "======================================"

# Auto-detect deployment type
if [ "$DEPLOYMENT_TYPE" == "auto" ]; then
    if docker ps | grep -q mayo; then
        DEPLOYMENT_TYPE="docker"
        print_msg "Detected Docker deployment"
    elif [ -f /tmp/mayo_api.pid ]; then
        DEPLOYMENT_TYPE="local"
        print_msg "Detected local deployment"
    else
        print_error "Could not detect deployment type"
        exit 1
    fi
fi

# Shutdown Docker deployment
shutdown_docker() {
    print_msg "Stopping Docker services..."

    cd "$SCRIPT_DIR"

    if [ -f docker-compose-full.yml ]; then
        docker-compose -f docker-compose-full.yml down
        print_msg "✓ Docker services stopped"
    else
        print_error "docker-compose-full.yml not found"
        exit 1
    fi

    # Optional: Remove volumes
    read -p "Remove data volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f docker-compose-full.yml down -v
        print_msg "✓ Volumes removed"
    fi
}

# Shutdown local deployment
shutdown_local() {
    print_msg "Stopping local API server..."

    if [ -f /tmp/mayo_api.pid ]; then
        PID=$(cat /tmp/mayo_api.pid)

        if ps -p $PID > /dev/null; then
            print_msg "Sending SIGTERM to process $PID..."
            kill -TERM $PID

            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! ps -p $PID > /dev/null; then
                    print_msg "✓ Process stopped gracefully"
                    rm /tmp/mayo_api.pid
                    return 0
                fi
                sleep 1
            done

            # Force kill if still running
            if ps -p $PID > /dev/null; then
                print_msg "Forcing process shutdown..."
                kill -9 $PID
                rm /tmp/mayo_api.pid
                print_msg "✓ Process force-stopped"
            fi
        else
            print_msg "Process already stopped"
            rm /tmp/mayo_api.pid
        fi
    else
        print_msg "No PID file found"
    fi
}

# Main shutdown
main() {
    case $DEPLOYMENT_TYPE in
        docker)
            shutdown_docker
            ;;
        local)
            shutdown_local
            ;;
        *)
            print_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            print_msg "Usage: $0 [docker|local]"
            exit 1
            ;;
    esac

    print_msg "======================================"
    print_msg "Shutdown complete!"
    print_msg "======================================"
}

main
