"""Performs Mask R-CNN inference to segment PV modules in IR video frames.

This module runs inference of Mask R-CNN on IR video frames to segment PV
modules. Configuration settings are defined in `configs.py`, e.g. the minimum
detection confidence and weights file. The Mask R-CNN model can be trained
with the `train.ipynb` Ipython notebook and a suitable training dataset.
"""

import os
import glob
import csv
import logging
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the TensorFlow 2 version of Mask R-CNN
from extractor.segmentation.Mask_RCNN_TF2.mrcnn import model as modellib
from extractor.common import Capture, delete_output
from extractor.segmentation.configs import PVConfigIR, PVConfigRGB


# Configure TensorFlow 2 to use memory growth for GPUs
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            logging.info(f"Memory growth enabled for GPU device: {device}")
        logging.info(f"Found {len(physical_devices)} GPU(s) available")
    except RuntimeError as e:
        logging.warning(f"GPU configuration failed: {e}")
else:
    logging.warning("No GPU devices found. Running on CPU.")

# Disable eager execution as required by Mask R-CNN TF2 port
tf.compat.v1.disable_eager_execution()

logger = logging.getLogger(__name__)


def draw_masks(image, masks, alpha=0.6):
    """Draw colored masks on an image with the specified transparency."""
    if masks.shape[-1] > 0:
        for mask in np.split(masks, masks.shape[-1], axis=-1):
            image_masked = np.copy(image)
            mask = mask.squeeze()
            color = list(np.random.choice(range(256), size=3))
            for c in range(image.shape[-1]):
                image_masked[:, :, c] = np.where(
                    mask == 1, color[c], image[:, :, c])
            image = cv2.addWeighted(image, alpha, image_masked, 1.0-alpha, 0.0)
    return image


def save(frame, frame_name, result, output_dir, videowriter):
    """Save detection results as masks and metadata."""
    # Add frame with masks to output video
    frame_preview = draw_masks(frame, result["masks"], alpha=0.6)
    videowriter.write(frame_preview)

    # Write masks as PNG files
    mask_path_extended = os.path.join(output_dir, "masks", frame_name)
    os.makedirs(mask_path_extended, exist_ok=True)
    if result["masks"].shape[-1] > 0:
        for mask_id, mask in enumerate(
                np.split(result["masks"], result["masks"].shape[-1], axis=-1)):
            mask = mask.squeeze().astype(np.uint8)
            mask *= 255
            mask_file = os.path.join(
                mask_path_extended, "mask_{:06d}.png".format(mask_id))
            cv2.imwrite(mask_file, mask)

    # Write detection metadata (ROIs, scores, class IDs) in CSV file
    roi_file = os.path.join(output_dir, "rois", "{}.csv".format(frame_name))
    result_subset = {k: v
        for k, v in result.items()
        if k in ["rois", "class_ids", "scores"]}
    with open(roi_file, "w", newline='') as f:
        csvriter = csv.writer(f, delimiter=',')
        for roi, class_id, score in zip(
                result_subset["rois"],
                result_subset["class_ids"],
                result_subset["scores"]):
            csvriter.writerow([*roi, class_id, score])


def run(frames_root, output_dir, ir_or_rgb, gpu_count, images_per_gpu, 
    detection_min_confidence, weights_file_ir, weights_file_rgb, output_video_fps):
    """Run Mask R-CNN inference on a set of frames."""
    delete_output(output_dir)

    # Create output paths
    for p in ["masks", "rois"]:
        os.makedirs(os.path.join(output_dir, p), exist_ok=True)

    # Configure inference settings
    if ir_or_rgb == "ir":
        inference_config = PVConfigIR()
    else:
        inference_config = PVConfigRGB()
    inference_config.GPU_COUNT = gpu_count
    inference_config.IMAGES_PER_GPU = images_per_gpu
    inference_config.DETECTION_MIN_CONFIDENCE = detection_min_confidence
    inference_config.__init__()  # Recompute batch size

    # Create model and load pretrained weights
    model = modellib.MaskRCNN(mode="inference",
                      config=inference_config,
                      model_dir="")

    # Select appropriate weights file and input files
    if ir_or_rgb == "ir":
        weights_file = weights_file_ir
        frame_files = sorted(glob.glob(os.path.join(frames_root, "radiometric", "*.tiff")))
    else:
        weights_file = weights_file_rgb
        frame_files = sorted(glob.glob(os.path.join(frames_root, "rgb", "*.jpg")))

    logger.info(f"Loading weights from {weights_file}")
    model.load_weights(weights_file, by_name=True)

    cap = Capture(frame_files, ir_or_rgb, mask_files=None)
    step_idx = 0

    batch_size = model.config.BATCH_SIZE
    frames_batch = []
    frame_names_batch = []

    # Set up video writer for preview output
    video_shape = (cap.img_w, cap.img_h)
    video_path = os.path.join(output_dir, "preview.avi")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videowriter = cv2.VideoWriter(video_path, fourcc, output_video_fps, video_shape)

    pbar = tqdm(total=len(frame_files))
    while True:
        frame, _, frame_name, _ = cap.get_next_frame(preprocess=True)
        if frame is None:
            break

        # Prepare the frame for inference
        if ir_or_rgb == "ir":
            frame = np.stack((frame, frame, frame), axis=2)  # Make 3-channel image
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR -> RGB

        frames_batch.append(frame)
        frame_names_batch.append(frame_name)

        # Handle last batch (pad with zeros if smaller than batch size)
        if step_idx == len(frame_files) - 1:
            orig_batch_len = len(frames_batch)
            for _ in range(batch_size - orig_batch_len):
                frames_batch.append(np.zeros_like(frames_batch[0]))
            results = model.detect(frames_batch, verbose=0)  # Model inference
            results = results[:orig_batch_len]
            frames_batch = frames_batch[:orig_batch_len]
            for frame, frame_name, result in zip(
                    frames_batch, frame_names_batch, results):
                if ir_or_rgb == "rgb":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                save(frame, frame_name, result, output_dir, videowriter)
            break

        # Run inference on a batch of frames
        if step_idx % batch_size == batch_size - 1:
            results = model.detect(frames_batch, verbose=0)
            for frame, frame_name, result in zip(
                    frames_batch, frame_names_batch, results):
                if ir_or_rgb == "rgb":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                save(frame, frame_name, result, output_dir, videowriter)
            frames_batch = []
            frame_names_batch = []

        pbar.update(1)
        step_idx += 1

    pbar.close()
    videowriter.release()
    logger.info(f"Segmentation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Mask R-CNN inference on PV module frames.")
    parser.add_argument("--frames_root", required=True, help="Root directory containing input frames.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output results.")
    parser.add_argument("--ir_or_rgb", choices=["ir", "rgb"], required=True, help="Dataset type: 'ir' or 'rgb'.")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use for inference.")
    parser.add_argument("--images_per_gpu", type=int, default=2, help="Number of images per GPU.")
    parser.add_argument("--detection_min_confidence", type=float, default=0.5, help="Minimum detection confidence.")
    parser.add_argument("--weights_file_ir", required=False, help="Path to IR weights file.")
    parser.add_argument("--weights_file_rgb", required=False, help="Path to RGB weights file.")
    parser.add_argument("--output_video_fps", type=int, default=30, help="FPS for output video.")

    args = parser.parse_args()

    run(args.frames_root, args.output_dir, args.ir_or_rgb,
        args.gpu_count, args.images_per_gpu,
        args.detection_min_confidence,
        args.weights_file_ir, args.weights_file_rgb,
        args.output_video_fps)