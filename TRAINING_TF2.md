# Training with TensorFlow 2

This document provides instructions for training PV-Hawk models with TensorFlow 2.

## Setup

The training notebook `extractor/segmentation/train.ipynb` has been updated to use TensorFlow 2 and the Mask_RCNN_TF2 implementation.

### Prerequisites

1. Activate the TensorFlow 2 environment:
   ```bash
   source .vevn_tf2/bin/activate
   ```

2. Install Mask_RCNN_TF2 (already done during migration):
   ```bash
   cd extractor/segmentation/Mask_RCNN_TF2
   pip install -e .
   ```

## Key Changes from TF1 to TF2

1. **Updated imports**: All imports now reference `Mask_RCNN_TF2` instead of `Mask_RCNN`
2. **GPU memory growth**: Automatic GPU memory growth is enabled for better GPU utilization
3. **Eager execution disabled**: Required for Mask R-CNN compatibility
4. **Model paths**: All model and weight paths updated to use TF2 versions

## Training Process

The training notebook includes:

1. **Configuration**: Choose between RGB and IR training configurations
2. **Data loading**: Custom PVDataset class for loading PV module annotations
3. **Two-stage training**:
   - Stage 1: Train only the heads (classification and regression layers)
   - Stage 2: Fine-tune all layers with lower learning rate
4. **Validation**: Compute mAP and F1 scores at different IoU thresholds

## GPU Requirements

- NVIDIA GPU with CUDA support
- At least 6GB GPU memory (RTX 4060 or better recommended)
- NVIDIA Container Toolkit for Docker deployment

## Usage

1. Open the training notebook:
   ```bash
   jupyter notebook extractor/segmentation/train.ipynb
   ```

2. Configure your dataset paths in the configuration cells
3. Run cells sequentially to train the model
4. Monitor training progress and validation metrics

## Model Outputs

Trained models are saved in the `Mask_RCNN_TF2/logs` directory with timestamps and can be used for inference with the updated `inference.py` script.