Contributing
============

PV Hawk is non-profit and open source. As such we rely on your contribution to make this project even more awesome. Below I will describe some open research avenues you may want to look into. If you want to work on any of these or have own ideas on how to improve the project, feel free to open an issue on GitHub for further discussion.

PV module segmentation
----------------------

As mention in the :doc:`limitations`, PV module segmentation is currently robust for PV plants with row-based layout, but has issues with rooftop plants.


Use of visual videos in addition to IR videos
---------------------------------------------

Stability of the mapping step may be greatly enhanced by using visual videos instead of thermal IR videos. This is because visual videos provide a larger resolution, color information and a much lower dynamic range than IR videos. Furthermore, most computer vision algorithms used in PV Hawk were initially developed for the visual rather than thermal domain.
Successful inclusion of visual imagery requires accurate temporal synchronization and spatial registration with the IR video stream. This is in itself a sizeable but interesting engineering challenge.


Extension to EL/PL imaging
--------------------------

Apart from thermal IR imaging there exist other imaging techniques, such as electroluminescence (EL) or photoluminescence (PL) imaging, for the inspection of PV plants. An interesting line of research is the extension of PV Hawk to these imaging techniques.


Scaling to very large plants
----------------------------

In order to make PV Hawk useful for the processing of very large PV plants (larger than 100 MWp), some engineering challenges must be addressed. For instance, the processing pipeline must be scaled both vertically (use of multiple processor cores) and horizontally (use of multiple compute nodes). Furthermore, more advanced data sharding may be needed to deal with larger datasets.
