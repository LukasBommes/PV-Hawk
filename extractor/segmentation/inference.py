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

import extractor.segmentation.Mask_RCNN.mrcnn.model as modellib
from extractor.common import Capture, delete_output
from extractor.segmentation.configs import InferenceConfig

# Bugfix taken from:
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
# Moved this into Celery task, otherwise celery worker hangs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)


logger = logging.getLogger(__name__)


def draw_masks(image, masks, alpha=0.6):
    if masks.shape[-1] > 0:
        for mask in np.split(masks, masks.shape[-1], axis=-1):
            image_masked = np.copy(image)
            mask = mask.squeeze()
            color = list(np.random.choice(range(256), size=3))
            for c in range(image.shape[-1]):
                image_masked[:, :, c] = np.where(
                    mask == 1, color[c], image[:, :, c])
            alpha = 0.6
            image = cv2.addWeighted(image, alpha, image_masked, 1.0-alpha, 0.0)
    return image


def save(frame, frame_name, result, output_dir, videowriter):
    # add frame to output video
    frame_preview = draw_masks(frame, result["masks"], alpha=0.6)
    videowriter.write(frame_preview)

    # write masks as PNG files
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

    # write rois, scores and class_ids in CSV file
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


def run(frames_root, output_dir, output_video_fps):

    delete_output(output_dir)

    # create output paths
    for p in ["masks", "rois"]:
        os.makedirs(os.path.join(output_dir, p), exist_ok=True)

    inference_config = InferenceConfig()

    # create model and load pretrained weights
    model = modellib.MaskRCNN(mode="inference",
                      config=inference_config,
                      model_dir="")
    weights_file = model.config.WEIGHTS_FILE
    logger.info("Loading weights from {}".format(weights_file))
    model.load_weights(weights_file, by_name=True)

    frame_files = sorted(glob.glob(os.path.join(frames_root, "*.tiff")))
    cap = Capture(frame_files, mask_files=None)
    step_idx = 0

    batch_size = model.config.BATCH_SIZE
    frames_batch = []
    frame_names_batch = []

    # video writer
    video_shape = (cap.img_w, cap.img_h)
    video_path = os.path.join(output_dir, "preview.avi")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videowriter = cv2.VideoWriter(video_path, fourcc, output_video_fps, video_shape)

    pbar = tqdm(total=len(frame_files))
    while True:
        frame, _, frame_name, _ = cap.get_next_frame(preprocess=True)
        if frame is None:
            break
        frame = np.stack((frame, frame, frame), axis=2)  # make 3-channel image
        frames_batch.append(frame)
        frame_names_batch.append(frame_name)

        # handle last batch (pad with zeros if smaller than batch size)
        if step_idx == len(frame_files) - 1:
            orig_batch_len = len(frames_batch)
            for _ in range(batch_size - orig_batch_len):
                frames_batch.append(np.zeros_like(frames_batch[0]))
            results = model.detect(frames_batch, verbose=0)  # model inference
            results = results[:orig_batch_len]
            frames_batch = frames_batch[:orig_batch_len]
            for frame, frame_name, result in zip(
                    frames_batch, frame_names_batch, results):
                save(frame, frame_name, result, output_dir, videowriter)
            break

        # run inference on a batch of frames
        if step_idx % batch_size == batch_size - 1:
            results = model.detect(frames_batch, verbose=0)
            for frame, frame_name, result in zip(
                    frames_batch, frame_names_batch, results):
                save(frame, frame_name, result, output_dir, videowriter)
            frames_batch = []
            frame_names_batch = []

        pbar.update(1)
        step_idx += 1

    pbar.close()
    videowriter.release()
