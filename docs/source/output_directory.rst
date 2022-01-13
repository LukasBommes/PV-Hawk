Output Directory Structure
==========================

After completing a processing task a new subdirectory with results of that task will be created under the `work_dir` specified in the config file. E.g. after running the `segment_pv_modules` tasks, there will be a new directory called `segmented`. Whenever you rerun a task, the respective subdirectory is deleted automatically before that task starts.

Note, that the directory tree below only shows important files and subdirectories. We will briefly explain them below.

.. code-block:: text

  /workdir
    |-- splitted
    |    |-- timestamps.csv
    |    |-- gps
    |    |     |-- gps.json
    |    |     |-- gps.kml
    |    |-- preview
    |    |     |-- frame_000000.jpg
    |    |     |-- frame_000001.jpg
    |    |     |-- ...
    |    |-- radiometric
    |    |     |-- frame_000000.tiff
    |    |     |-- frame_000001.tiff
    |    |     |-- ...
    |-- segmented
    |    |-- preview.avi
    |    |-- ...
    |-- tracking
    |    |-- tracks_preview.avi
    |    |-- tracks.csv
    |-- patches
    |    |-- meta.pkl
    |    |-- preview
    |    |     |-- ...
    |    |-- radiometric
    |    |     |-- ...
    |-- mapping
    |    |-- cluster_000000
    |    |     |-- reconstruction.json
    |    |     |-- ...
    |    |-- cluster_000001
    |    |     |-- reconstruction.json
    |    |     |-- ...
    |    |-- ...
    |    |-- module_geolocations_refined.geojson
    |    |-- ...
    |-- patches_final
    |    |-- preview
    |    |     |-- ...
    |    |-- radiometric
    |    |     |-- 840d60c7-e634-45be-9043-48110873c8e4
    |    |     |     |-- frame_010401_mask_000010.tiff
    |    |     |     |-- frame_010402_mask_000009.tiff
    |    |     |     |-- frame_010403_mask_000007.tiff
    |    |     |     |-- ...
    |    |     |-- 05bee5f1-9d66-4ac7-aee4-c724d59663b1
    |    |     |     |-- ...
    |    |     |-- ...
    
.. rubric:: splitted (split_sequences) 

Contains the individual IR video frames as 16-bit TIFF images and additional 8-bit JPEG preview images. The `gps.json` files contains the longitude, latitude, and altitude (in this order) of each video frame. The `timestamps.csv` contains timestamps of each video frame.

.. rubric:: segmented (segment_pv_modules)

Contains results of the Mask R-CNN instance segmentation model applied to each video frame. Specifically, this directory contains binary segmentation masks of PV modules, bounding boxes and preview images. To validate correctness of the instance segmentation you can look at the preview video in `preview.avi`.

.. rubric:: tracking (track_pv_modules)

Contains the results of PV module tracking over subsequent video frames. The `tracks.csv` contains the frame name, mask name, tracking ID, and module center point in the image (x, y in pixels). Each PV module has a unique tracking ID that stays constant over subsequent video frame, in which the module is visible. To validate correctness of the module tracking you can look at the preview video in `tracks_preview.avi`.

.. rubric:: patches (crop_and_rectify_modules)

Contains the cropped and rectified image patches of each PV module. The `preview` directory contains 8-bit JPEG preview images and the `radiometric` directory the respective 16-bit TIFF images. For each PV module there is a directory named after the module's tracking ID, which contains the individual image patches showing the same module in subsequent video frames. The `meta.pkl` is a Python pickle file, containing additional information about each image patch, such as the image coordinates of the module's center point, the bounding quadrilateral that was fit to the module's segmentation mask, and the homography used to rectify the module image.

.. rubric:: mapping (prepare_opensfm, ..., refine_triangulation)

Contains the inputs and outputs of the tasks relating to the georeferencing of PV modules, which is performed using `OpenSfM <https://github.com/mapillary/OpenSfM>`_. For each cluster configured in the config file there is a subdirectory, which contains the OpenSfM dataset for that cluster and which has the structure described `here <https://opensfm.org/docs/dataset.html>`_. Most notably, each of these subdirectories contains the `reconstruction.json` file with the reconstructed camera poses and 3D map points produced by OpenSfM.
The main result stored in the mapping directory is a GeoJSON file `module_geolocations_refined.geojson`. This file follows the `GeoJSON specification <https://datatracker.ietf.org/doc/html/rfc7946>`_ and contains a feature collection of polygons and points for each PV module. The polygon resembles the longitude, latitude and altitude of the four corner points of the PV module in `WGS 84 coordinates <https://epsg.io/4326>`_. The point is the geocoordinate of the module's center point. Each polygon and each point have a `track_id` property, which is the tracking ID of the respective PV module.

.. rubric:: patches_final (reorganize_patches)

This directory has the same overall structure as the `patches` directory, but considers the merging of tracking IDs. Merged tracking IDs occur when the same PV module has two different tracking IDs, which can happen occasionally due to a tracking error. These duplicates are identified during the georeferencing procedure. If for example the trackings IDs `abc123` and `def456` belong to the same module, there will be only one directory in the `patches_final` directory named after the first tracking ID (`abc123`), which contains all module images of both subfolders `abc123` and `def456` from the `patches` directory.
