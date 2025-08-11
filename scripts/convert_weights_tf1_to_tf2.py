"""
Converts Mask R-CNN weights from TensorFlow 1.x format to TensorFlow 2.x format.
"""

import os
import sys
import h5py
import numpy as np
import tensorflow as tf

# Configure TensorFlow 2 to use memory growth for GPUs
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Disable eager execution as required by Mask R-CNN TF2 port
tf.compat.v1.disable_eager_execution()

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from extractor.segmentation.Mask_RCNN_TF2.mrcnn import model as modellib
from extractor.segmentation.configs import PVConfigIR, PVConfigRGB

def convert_weights_simple(source_weights_path, target_weights_path, model_type="ir"):
    """
    Simple weight conversion using TF1 compatibility mode.
    """
    print(f"Converting weights from {source_weights_path} to {target_weights_path}")
    
    # Create config and model
    if model_type == "ir":
        config = PVConfigIR()
    else:
        config = PVConfigRGB()
        
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 1
    config.__init__()  # Recompute batch size
    
    # Enable eager execution temporarily for weight loading
    tf.compat.v1.enable_eager_execution()
    
    try:
        # Create model in inference mode
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
        
        # Load the original weights using Keras compatibility
        print("Loading original weights...")
        model.keras_model.load_weights(source_weights_path, by_name=True, skip_mismatch=True)
        
        # Save in TF2 format
        print("Saving converted weights...")
        model.keras_model.save_weights(target_weights_path)
        
        print(f"Weights successfully converted and saved to {target_weights_path}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False
    finally:
        # Disable eager execution again
        tf.compat.v1.disable_eager_execution()
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert TF1 Mask R-CNN weights to TF2 format")
    parser.add_argument("--source", required=True, help="Source TF1 weights file path")
    parser.add_argument("--target", required=True, help="Target TF2 weights file path")
    parser.add_argument("--model_type", choices=["ir", "rgb"], default="ir", 
                        help="Model type (ir or rgb)")
    
    args = parser.parse_args()
    
    convert_weights_simple(args.source, args.target, args.model_type)