#!/bin/bash

# PV-Hawk Model Download Script
# Downloads pre-trained model weights for PV module detection

set -e

MODELS_DIR="models"
BASE_URL="https://github.com/LukasBommes/PV-Hawk/releases/download/v1.0.0"

echo "📥 Downloading PV-Hawk model weights..."
echo "======================================"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Download IR model weights
IR_MODEL="mask_rcnn_pv_modules_0120.h5"
if [ ! -f "$MODELS_DIR/$IR_MODEL" ]; then
    echo "Downloading IR model weights..."
    curl -L -o "$MODELS_DIR/$IR_MODEL" "$BASE_URL/$IR_MODEL"
    echo "✅ Downloaded: $IR_MODEL"
else
    echo "✅ Already exists: $IR_MODEL"
fi

# Download RGB model weights
RGB_MODEL="mask_rcnn_pv_modules_rgb_0059.h5"
if [ ! -f "$MODELS_DIR/$RGB_MODEL" ]; then
    echo "Downloading RGB model weights..."
    curl -L -o "$MODELS_DIR/$RGB_MODEL" "$BASE_URL/$RGB_MODEL"
    echo "✅ Downloaded: $RGB_MODEL"
else
    echo "✅ Already exists: $RGB_MODEL"
fi

# Download COCO pre-trained weights (optional)
COCO_MODEL="mask_rcnn_coco.h5"
if [ ! -f "$MODELS_DIR/$COCO_MODEL" ]; then
    echo "Downloading COCO pre-trained weights (optional)..."
    curl -L -o "$MODELS_DIR/$COCO_MODEL" "$BASE_URL/$COCO_MODEL"
    echo "✅ Downloaded: $COCO_MODEL"
else
    echo "✅ Already exists: $COCO_MODEL"
fi

echo ""
echo "🎉 Model download complete!"
echo "Models are now available in the $MODELS_DIR/ directory"
echo ""
echo "Available models:"
echo "- $IR_MODEL (for infrared/thermal images)"
echo "- $RGB_MODEL (for RGB/visual images)"
echo "- $COCO_MODEL (COCO pre-trained weights)"