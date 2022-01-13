Current Limitations
===================

Plant layouts
-------------

PV Drone Inspect was developed for large-scale ground-mounted PV plants with regular row-based layouts. In principle, plants with irregular layouts (rooftop systems) or regular non-row based layouts (floating PV, large rooftop arrays) can also be processed. However, we have not yet extensively tested PV Drone Inspect on such plants. If you still want to use PV Drone Inspect for irregular PV plants you have to consider some things.

For example, when inspecting large arrays of densely packed PV modules (as common on large rooftops), you need to record the array in multiple overlapping sweeps. For accurate reconstruction of these sweeps, high GPS accuracy, for instance provided by RTK-GPS, is required. Standard GPS will not suffice.

Another issue occurs with rooftop plants, where there are more obstructions than in a ground-mounted plant, e.g. windows, chimneys, piping, etc. In these cases PV module segmentation by Mask R-CNN is likely to fail as lots of false positives are generated. This is because we trained Mask R-CNN on a dataset containing only ground-mounted PV plants. Future versions of PV Drone Inspect should consider this by re-training the Mask R-CNN on a larger corpus of PV plants including rooftop plants. If you want to use PV Drone Inspect for such rooftop plants, you need to fine-tune Mask R-CNN yourself for the time being as described in :doc:`finetune_segmentation`.

OpenSfM reconstruction failures
-------------------------------

Scanning individual PV plant rows from low altitude is a challenging scenario for reconstruction with openSfM. Furthermore, we solely use IR imagery, which has a lower resolution and is more blurry than visual imagery, making the reconstruction more difficult.
Thus, the reconstruction procedure can fail leading to corrupted 3D reconstructions and PV module locations. We noted that this occurs quite frequently and requires tuning of the settings for the OpenSfM reconstruction. Splitting a longer video sequence into smaller clusters of at most 2000 images can further improve robustness of the reconstruction procedure. A long sequence should also be split into clusters whenever there are discontinuities in the video, e.g. due to battery changes, or sudden movements. This prevent the reconstruction from failing at those video frames.
