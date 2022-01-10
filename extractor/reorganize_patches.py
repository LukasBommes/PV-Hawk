"""Finalize dataset.

Creates final output directories and files.
Creates a patches directory containing patches only for those modules
which could be reconstructed in OpenSfM. It further considers the merger
of duplicate modules during OpenSfM reconstruction.
"""

import os
import glob
import shutil
import pickle
import logging
from tqdm import tqdm

from extractor.common import delete_output


logger = logging.getLogger(__name__)


def run(mapping_root, patches_root, output_dir):

    delete_output(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # handle modules merged during 3D reconstruction
    merged_modules = pickle.load(open(os.path.join(mapping_root, "merged_modules.pkl"), "rb"))
    module_corners = pickle.load(open(os.path.join(mapping_root, "module_corners.pkl"), "rb"))

    # copy patches from patches to final/patches and merge the patches of merged modules
    logger.info("Reorganizing patches")
    for track_id in tqdm(module_corners.keys()):
        dst = os.path.join(output_dir, "radiometric", track_id)
        os.makedirs(dst, exist_ok=True)

        # get modules the current module is (possibly) merged with
        source_modules = set([track_id])
        for modules in merged_modules:
            if track_id in modules:
                source_modules |= set(modules)
        
        # copy patches of the module and the ones it is merged with to destintation folder        
        for track_id_ in source_modules:
            patch_files = sorted(glob.glob(os.path.join(patches_root, "radiometric", track_id_, "*")))           
            
            for patch_file in patch_files:
                shutil.copy2(patch_file, dst)
                #print(patch_file, dst)