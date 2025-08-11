# PV-Hawk Docker Usage Guide

This guide provides comprehensive instructions for running PV-Hawk using Docker and Docker Compose with TensorFlow 2 and GPU support.

## 📋 Prerequisites

### System Requirements
- **Docker** version 19.03+ with GPU support
- **NVIDIA Container Toolkit** (for GPU acceleration)
- **NVIDIA GPU** with CUDA support
- **Linux, Windows with WSL2, or macOS with Docker Desktop**

### GPU Setup
1. **Install NVIDIA Container Toolkit:**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Test GPU access:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

## 🚀 Quick Start with Docker Compose

### 1. Prepare Your Environment

```bash
# Clone the repository (if not already done)
git clone https://github.com/mustafa-meco/PV-Hawk.git
cd PV-Hawk

# Checkout the TensorFlow 2 branch
git checkout tensorflow2-migration

# Copy environment template
cp .env.example .env

# Edit environment variables to match your setup
nano .env
```

### 2. Prepare Your Data Structure

Create the following directory structure:
```
your-project/
├── data/                    # Input videos and raw data
├── workdir/                 # Processing workspace
│   └── config.yml          # Your processing configuration
├── config/                  # Configuration files
├── calibration/            # Camera calibration parameters
├── models/                 # Model weights (.h5 files)
└── output/                 # Final results
```

### 3. Configure Your Processing

```bash
# Copy and edit the configuration
cp config/config.yml.example workdir/config.yml
nano workdir/config.yml
```

### 4. Build and Run

```bash
# Build the Docker image
docker-compose build

# Run the processing pipeline
docker-compose up pv-hawk

# Or run in detached mode
docker-compose up -d pv-hawk
```

## 🛠️ Manual Docker Commands

### Build the Image
```bash
docker build -t pv-hawk-tf2:latest .
```

### Basic Run Command
```bash
docker run --gpus all --rm -it \
  -v $(pwd)/workdir:/workdir \
  -v $(pwd)/data:/data \
  -v $(pwd)/config:/config \
  -v $(pwd)/models:/models \
  pv-hawk-tf2:latest python3 main.py /workdir
```

### Advanced Run with All Mounts
```bash
docker run --gpus all --rm -it \
  --name pv-hawk-processor \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -v $(pwd)/workdir:/workdir \
  -v $(pwd)/data:/data \
  -v $(pwd)/config:/config \
  -v $(pwd)/calibration:/calibration \
  -v $(pwd)/models:/models \
  -v $(pwd)/output:/output \
  pv-hawk-tf2:latest python3 main.py /workdir
```

### Interactive Shell Access
```bash
docker run --gpus all --rm -it \
  -v $(pwd)/workdir:/workdir \
  -v $(pwd)/data:/data \
  pv-hawk-tf2:latest /bin/bash
```

### Run Specific Inference Task
```bash
docker run --gpus all --rm -it \
  -v $(pwd)/data:/data \
  -v $(pwd)/models:/models \
  pv-hawk-tf2:latest \
  python3 extractor/segmentation/inference.py \
  --frames_root /data/splitted \
  --output_dir /data/segmented \
  --ir_or_rgb ir \
  --weights_file_ir /models/mask_rcnn_pv_modules_0120.h5
```

## 🔧 Development Mode

For development with Jupyter Lab:

```bash
# Start Jupyter Lab service
docker-compose --profile development up jupyter

# Access at http://localhost:8888
```

## 📂 Directory Structure and Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./workdir` | `/workdir` | Processing workspace with config.yml |
| `./data` | `/data` | Input videos and raw data |
| `./config` | `/config` | Configuration templates |
| `./calibration` | `/calibration` | Camera calibration parameters |
| `./models` | `/models` | Pre-trained model weights |
| `./output` | `/output` | Final processing results |

## ⚙️ Environment Variables

Key environment variables (set in `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |
| `TF_FORCE_GPU_ALLOW_GROWTH` | `true` | Allow GPU memory growth |
| `TF_CPP_MIN_LOG_LEVEL` | `1` | TensorFlow logging level |
| `DATA_DIR` | `./data` | Host data directory |
| `WORK_DIR` | `./workdir` | Host working directory |
| `MODELS_DIR` | `./models` | Host models directory |

## 🔍 Monitoring and Debugging

### View Logs
```bash
# View logs in real-time
docker-compose logs -f pv-hawk

# View specific container logs
docker logs pv-hawk-processor
```

### Monitor GPU Usage
```bash
# Monitor GPU while processing
nvidia-smi -l 1
```

### Check Container Health
```bash
# Check health status
docker-compose ps

# Inspect container
docker inspect pv-hawk-processor
```

### Debug Interactive Mode
```bash
# Enter running container
docker exec -it pv-hawk-processor /bin/bash

# Test TensorFlow GPU
docker exec -it pv-hawk-processor python3 -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Check Docker GPU support
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Out of GPU Memory**
   - Reduce `images_per_gpu` in config.yml
   - Set `TF_FORCE_GPU_ALLOW_GROWTH=true`

3. **Permission Errors**
   ```bash
   # Fix directory permissions
   sudo chown -R $USER:$USER ./workdir ./data ./output
   ```

4. **Docker Build Fails**
   ```bash
   # Clean build with no cache
   docker build --no-cache -t pv-hawk-tf2:latest .
   
   # Check disk space
   docker system df
   docker system prune
   ```

### Performance Optimization

1. **GPU Memory Settings**
   - Monitor with `nvidia-smi`
   - Adjust batch sizes in configuration
   - Use `TF_FORCE_GPU_ALLOW_GROWTH=true`

2. **CPU Optimization**
   - Set OpenSfM processes based on CPU cores
   - Use SSD storage for better I/O performance

3. **Docker Performance**
   - Use volumes instead of bind mounts for better performance
   - Allocate sufficient Docker resources

## 📝 Example Workflows

### Process IR Video Sequence
```bash
# 1. Set up data
mkdir -p workdir/group_001/videos
cp your_ir_video.tiff workdir/group_001/videos/

# 2. Create config
cat > workdir/config.yml << EOF
plant_name: "IR Processing Test"
groups:
- cam_params_dir: /calibration
  clusters:
  - cluster_idx: 0
    frame_idx_start: 0
    frame_idx_end: 1000
tasks:
- split_sequences
- segment_pv_modules
EOF

# 3. Run processing
docker-compose up pv-hawk
```

### Process RGB Video with Full Pipeline
```bash
# Complete config with all tasks
cp config/config.yml.example workdir/config.yml

# Run full pipeline
docker-compose up pv-hawk
```

## 📊 Output Structure

After processing, your workdir will contain:
```
workdir/
├── config.yml
└── group_001/
    ├── splitted/          # Extracted frames
    ├── segmented/         # PV module segmentation results
    ├── tracking/          # Module tracking data
    ├── quadrilaterals/    # Module corner estimates
    ├── mapping/           # 3D reconstruction data
    └── patches/           # Individual module crops
```

## 🔄 Updates and Maintenance

### Update the Image
```bash
# Pull latest changes
git pull origin tensorflow2-migration

# Rebuild image
docker-compose build --no-cache

# Clean up old images
docker image prune -f
```

### Backup Processing Results
```bash
# Create backup
tar -czf pv-hawk-results-$(date +%Y%m%d).tar.gz workdir/ output/

# Restore from backup
tar -xzf pv-hawk-results-YYYYMMDD.tar.gz
```