#!/bin/sh
set -e

# Setup Mask R-CNN
cd /pvextractor/extractor/segmentation/Mask_RCNN
python setup.py install
cd /pvextractor

# Setup OpenSfM
cd /pvextractor/extractor/mapping/OpenSfM
python setup.py build
cd /pvextractor

exec "$@"
