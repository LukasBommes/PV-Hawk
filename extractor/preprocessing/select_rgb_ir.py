"""Select whether to use RGB or IR frames in further steps of the pipeline.
This creates a json file named `selected_ir_rgb.json`."""

import os
import json


def run(frames_root, which=None):
    out_file = os.path.join(frames_root, "selected_ir_rgb.json")

    try:
        os.remove(out_file)
    except FileNotFoundError:
        pass

    with open(out_file, "w") as file:
        json.dump({
            "selected": which
        }, file)
    