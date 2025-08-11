#!/bin/bash

# PV-Hawk Docker Setup Script
# This script helps set up the environment for running PV-Hawk with Docker

set -e

echo "🚀 PV-Hawk Docker Setup"
echo "======================="

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  This script is designed for Linux. For Windows/macOS, please follow the manual setup in DOCKER_USAGE.md"
    exit 1
fi

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker found: $(docker --version)"

# Check Docker Compose installation
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker Compose found: $(docker-compose --version)"

# Check NVIDIA Docker support
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    
    # Test NVIDIA Container Toolkit
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA Container Toolkit is working"
    else
        echo "⚠️  NVIDIA Container Toolkit may not be properly configured"
        echo "   Please install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
else
    echo "⚠️  No NVIDIA GPU detected. The system will run on CPU only."
fi

# Create directory structure
echo "📁 Creating directory structure..."

directories=(
    "data"
    "workdir"
    "config"
    "calibration" 
    "models"
    "output"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "   Created: $dir/"
    else
        echo "   Exists: $dir/"
    fi
done

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "   Please edit .env to match your system paths"
else
    echo "✅ .env file already exists"
fi

# Copy config file if it doesn't exist
if [ ! -f "workdir/config.yml" ]; then
    cp config/config.yml.example workdir/config.yml
    echo "✅ Created config.yml from template"
    echo "   Please edit workdir/config.yml for your processing needs"
else
    echo "✅ config.yml already exists"
fi

# Set proper permissions
echo "🔧 Setting directory permissions..."
chmod -R 755 data workdir config calibration models output
echo "✅ Permissions set"

# Test docker-compose configuration
echo "🧪 Testing Docker Compose configuration..."
if docker-compose config --quiet; then
    echo "✅ Docker Compose configuration is valid"
else
    echo "❌ Docker Compose configuration has errors"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file to set your data paths"
echo "2. Edit workdir/config.yml for your processing pipeline"
echo "3. Place your video files in the data/ directory"
echo "4. Download model weights to the models/ directory"
echo "5. Run: docker-compose build"
echo "6. Run: docker-compose up pv-hawk"
echo ""
echo "For detailed usage instructions, see DOCKER_USAGE.md"