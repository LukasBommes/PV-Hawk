How PV Hawk Works
=================

After acquisition with the drone, IR videos of a PV plant are split into individual frames and the GPS trajectory of the drone is extracted and interpolated. Following Bommes et al. [2], PV modules are segmented by Mask R-CNN [29], tracked over subsequent frames, extracted and stored to disk. To georeference PV modules, a subset of keyframes is selected based on travelled GPS distance and visual overlap. Subsequently, a georeferenced 3D reconstruction of the PV plant is obtained by incremental SfM (implemented in OpenSfM) alongside the 6-DOF camera pose of each keyframe. This requires calibrated camera parameters, which are obtained beforehand. The known keyframe poses are then used to triangulate observed PV modules into the 3D reconstruction, yielding the desired module geocoordinates.


Inputs to PV Hawk are IR video frames with the corresponding coarse GPS position (latitude, longitude, and optionally altitude) of the drone. Outputs are multiple cropped and rectified IR image patches per PV module and geocoordinates of the four corners of each PV module. PV Hawk first segments all PV modules in each video frame with a Mask R-CNN instance segmentation model. Each module is then tracked over subsequent frames. The corresponding image regions are cropped, rectified, and stored to disk with a unique tracking ID. 

For the localization of PV modules in the PV plant a flexible, layout-independent, and fully automated solution based on structure from motion is provided. The `OpenSfM <https://opensfm.org>`_ structure from motion library is used to reconstruct the 6-DOF camera trajectory.


 Since the coarse GPS trajectory of the drone is known the camera trajectory is reconstructed in geocoordinates. Based on the known camera coordinates, PV module corners are triangulated into the 3D reconstruction and refined with an iterative graph optimization, yielding absolute geocoordinates for each PV module (see fig. 3).

.. image:: images/overview.png


- reference to output dataset strcuture and config file reference
