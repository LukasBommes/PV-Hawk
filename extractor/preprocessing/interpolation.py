"""Perform piecewise linear interpolation of the GPS trajectory 
to match GPS measurement rate and video frame rate."""

import os
import json
import logging
import numpy as np

from extractor.gps import gps_to_ltp, gps_from_ltp, interpolate_gps


logger = logging.getLogger(__name__)


def run(frames_root):

    try:
        gps = json.load(open(os.path.join(
            frames_root, "gps", "gps_orig.json"), "r"))
    except FileNotFoundError:
        gps = json.load(open(os.path.join(
            frames_root, "gps", "gps.json"), "r"))

        assert len(gps) > 0
        assert len(gps[0]) == 2 or len(gps[0]) == 3

        json.dump(gps, open(os.path.join(
            frames_root, "gps", "gps_orig.json"), "w"))
    
    assert len(gps) > 0
    assert len(gps[0]) == 2 or len(gps[0]) == 3

    # interpolate GPS trajectory
    gps = np.array(gps)
    if gps.shape[-1] == 2:
        gps = np.insert(
            gps, 2, np.zeros(len(gps)), axis=1)
    gps, origin = gps_to_ltp(gps)
    gps = interpolate_gps(gps)
    gps = gps_from_ltp(gps, origin)
    gps = gps.tolist()

    json.dump(gps, open(os.path.join(
        frames_root, "gps", "gps.json"), "w"))