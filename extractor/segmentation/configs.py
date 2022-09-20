import numpy as np

try:
    from extractor.segmentation.Mask_RCNN.mrcnn.config import Config
except ModuleNotFoundError:
    from Mask_RCNN.mrcnn.config import Config


class PVConfig(Config):
    # Give the configuration a recognizable name
    NAME = "pv_modules"

    # Path to PV module dataset
    DATASET_TRAIN_PATH = "/pv_segmentation_dataset/train"
    DATASET_VAL_PATH = "/pv_segmentation_dataset/val"

    # Path for saving model weights and checkpoints during training
    MODEL_DIR = "/pvextractor/extractor/segmentation/Mask_RCNN/logs"

    # Path to MS COCO pretrained weights
    COCO_MODEL_PATH = "/pvextractor/extractor/segmentation/Mask_RCNN/mask_rcnn_coco.h5"
    
    # set to False to ignore partially visible (truncated) PV modules
    USE_TRUNCATED_MODULES = False
    
    # Train on 1 GPU and 2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + pv

    # Maximum number of ground truth instances to use in one image
    # TODO: set this to the maximum value in train/test set
    MAX_GT_INSTANCES = 200

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 200

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small
    
    
class PVConfigIR(PVConfig):
    # whether dataset contains monochrome 16-bit IR 
    # images ("ir") or 8-bit visual RGB images ("rgb")
    DATASET_MODE = "ir"
    
    # Batch size (in combination with GPU_COUNT)
    IMAGES_PER_GPU = 2
    
    # Images are resized and padded with zeros to [max_dim, max_dim] 
    # during training and prediction.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640
    
    # Image mean (RGB) computed from training set
    MEAN_PIXEL = np.array([129.4, 129.4, 129.4])
    
    
class PVConfigRGB(PVConfig):
    # whether dataset contains monochrome 16-bit IR 
    # images ("ir") or 8-bit visual RGB images ("rgb")
    DATASET_MODE = "rgb"
    
    # Batch size (in combination with GPU_COUNT)
    IMAGES_PER_GPU = 1
    
    # Images are resized and padded with zeros to [max_dim, max_dim] 
    # during training and prediction.
    IMAGE_MIN_DIM = 720
    IMAGE_MAX_DIM = 1280
    
    # Image mean (RGB) computed from training set
    MEAN_PIXEL = np.array([102.99604791, 108.6435939 , 125.23814966])
