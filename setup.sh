#!/bin/bash
# Quick start script for setting up the environment

echo "========================================="
echo "Mispronunciation Detection Setup"
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data/sample
mkdir -p logs
mkdir -p checkpoints
mkdir -p output
echo "✓ Directories created"
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run examples:"
echo "   python examples/run_examples.py"
echo ""
echo "3. Prepare your dataset (see README.md)"
echo ""
echo "4. Start training:"
echo "   python train.py --train_data data/train.json --val_data data/val.json"
echo ""
echo "========================================="
