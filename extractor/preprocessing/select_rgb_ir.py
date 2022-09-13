"""Select whether to use RGB or IR frames in further steps of the pipeline.
This creates a symlink named "images", either from "radiometric" or "rgb" folder."""

import os
import logging


logger = logging.getLogger(__name__)


def run(frames_root, which=None):

    if which == "rgb":
        src = os.path.join(frames_root, "rgb")
    elif which == "ir":
        src = os.path.join(frames_root, "radiometric")

    dst = os.path.join(frames_root, "images")

    try:
        os.unlink(dst)
    except OSError:
        pass

    os.symlink(src, dst)

    logger.info("Selected {} frames for further processing.".format(which.upper()))
    