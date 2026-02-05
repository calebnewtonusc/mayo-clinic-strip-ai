#!/bin/bash
# Setup script for Mayo Clinic STRIP AI project

echo "======================================"
echo "Mayo Clinic STRIP AI - Environment Setup"
echo "======================================"

# Check Python version
echo -e "\nChecking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\nCreating virtual environment..."
    python3 -m venv venv
else
    echo -e "\nVirtual environment already exists."
fi

# Activate virtual environment
echo -e "\nActivating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\nUpgrading pip..."
pip install --upgrade pip

# Install dependencies
echo -e "\nInstalling dependencies..."
pip install -r requirements.txt

# Verify installation
echo -e "\nVerifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"

# Create necessary directories
echo -e "\nCreating directory structure..."
mkdir -p data/{raw,processed,augmented,splits}/{CE,LAA}
mkdir -p experiments/{logs,checkpoints,results}
mkdir -p notebooks
mkdir -p scripts

echo -e "\n======================================"
echo "Setup complete!"
echo "======================================"
echo -e "\nNext steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Place your data in data/raw/{CE,LAA}/"
echo "3. Run data exploration: jupyter notebook notebooks/01_exploratory_data_analysis.ipynb"
echo "4. Preprocess data: python scripts/preprocess_data.py"
echo "5. Train model: python train.py"
