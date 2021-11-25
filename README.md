# PV Mapper

This is the implementation of the PV Mapper presented in the paper ["Computer Vision Tool for Detection, Mapping and Fault Classification of PV Modules in Aerial IR Videos"](https://arxiv.org/abs/2106.07314).

Its aim is to semi-automatically detect PV modules in aerial thermal infrared videos acquired by a drone. Detected modules are extracted from each individual video frame and associated with a plant ID that is manually provided.

The following diagram shows the components of the extraction tool.

![Overview diagram of the tool](overview.png)

Output is a directory containing multiple IR image patches for each PV module.

If you use PV-Mapper in your own research please consider citing us ([bibtex below](#citation)).

## Prerequisites

To run the PV Mapper you need a machine running Ubuntu 18.04 LTS / 20.04 LTS and a NVidia CUDA-compatible GPU with the latest NVidia drivers installed. Furthermore, you need to install [Docker CE](https://docs.docker.com/engine/install/ubuntu/) and the [Nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker).


## Installation

#### Step 1: Get source code

Clone the Git repository to your machine
```
git clone https://github.com/LukasBommes/PV-Mapper
```

#### Step 2: Download Mask R-CNN model files

The tool uses a pretrained [Mask R-CNN](https://github.com/matterport/Mask_RCNN) for PV module detection. Download the pretrained Mask R-CNN model weights from [here](https://drive.google.com/file/d/1F0GiR8QpKZEHV-4wtfbPeE5dvIiOeIG3/view?usp=sharing), extract the zip archive and place the folder "pv_modules20210521T1611" under `extractor/segmentation/Mask_RCNN/logs`. The resulting directory structure should look like follows:
```
|-- extractor/segmentation/Mask_RCNN/logs
                                        |-- pv_modules20210521T1611
                                            |-- events.out.tfevents.1621613480.27049cbf8e56
                                            |-- events.out.tfevents.1621782508.27049cbf8e56
                                            |-- mask_rcnn_pv_modules_0060.h5
                                            |-- mask_rcnn_pv_modules_0120.h5
```

#### Step 3: Build Docker image

We user Docker to provide a consistent environment for the PV-Mapper. When building the provided Docker image all required dependencies, e.g., Python, CUDA, Tensorflow, and OpenSfM, are installed and configured automatically. There are two ways to use the Docker image: A) building the image from the provided Dockerfile, or B) load a prebuilt image.

Note, that you need to build/load the image only once. Afterwards, you can run the Docker image as specified in the [usage section](#step-1:-run-docker-image).

##### Variant A: Build image from source

To build the Docker image from the provided Dockerifle run the following command from the root directory of PV Mapper
```
sudo docker build . --tag=pvextractor-geo
```

##### Variant B: Load prebuild image

Alternatively, you can download a prebuilt Docker image from [here](https://drive.google.com/file/d/1ksjtbYPbpkMeZbChfqtSVkXBKHAq9Nbo/view?usp=sharing). Place the tar archive in the project's root directory an load the Docker image by executing
```
sudo docker load < pvextractor-geo.tar
```
Note, that the image was built on a machine with Ubuntu 20.04 LTS. Transferability to other operating systems is not guaranteed. If you run into issue with the prebuild image, please build the image from source as specified above.


## Drone and Camera Setup



video format: 16-bit TIFF stack


#### Record IR Videos


#### Calibrate Camera



## Usage

#### Step 1: Run Docker image

You have to run the PV Mapper in an interactive terminal session inside the Docker image that you built in the previous steps. Before doing so, make sure access control of your machine's X server is disabled by running
```
xhost +
```
This enables graphical output from the Docker container to be forwarded to your machine. Note, that you have to repeat this step every time you rebooted your machine.

To run the interactive terminal session inside the Docker container run the following command from the project's root directory
```
sudo docker run -it \
    --ipc=host \
    --env="DISPLAY" \
    --gpus=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$(pwd)":/pvextractor \
    -v /storage-2:/storage-2 \
    -v /home/lukas/HI-ERN-2020/PV-Segmentation-Dataset/data:/pv_segmentation_dataset \
    -p "8888:8888" \
    pvextractor-geo \
    bash
```
You can omit the `-p "8888:8888"` option if you do not plan to use jupyter lab inside the container. Jupyter labe is for instance needed if you want to fine-tune the Mask R-CNN model contained in this tool.

If you encounter a "permission denied" error make the entrypoint script executable by running
```
chmod +x docker-entrypoint.sh
```
in the project root.


#### Step 2: Place IR video files


#### Step 3: Create config file

The config file specifies details of the PV plant, the input videos and the plant layout. An example is shown below.

```
---
plant_name: Plant A
work_dir: /storage-2/pvextractor-georeferencing/Plant_A/workdir
video_dir: /storage-2/pvextractor-georeferencing/Plant_A/
groups:
- name: Sector_1
  video_fps: 8.0
  row_orientation: horizontal
  cam_params_dir: calibration/camera_8hz/parameters
  clusters:
  - cluster_idx: 0
    frame_idx_start: 205
    frame_idx_end: 1147
  - cluster_idx: 1
    frame_idx_start: 2021
    frame_idx_end: 2838
  - cluster_idx: 2
    frame_idx_start: 2905
    frame_idx_end: 3397
  settings:
    triangulate_pv_modules:
      max_module_depth: 30
    refine_triangulation:
      max_module_depth: 30

- name: Sector_2
  video_fps: 8.0
  row_orientation: horizontal
  cam_params_dir: calibration/camera_8hz/parameters
  clusters:
  - cluster_idx: 0
    frame_idx_start: 208
    frame_idx_end: 2512
  settings:
    prepare_opensfm:
      frame_selection_gps_distance: 0.75
    refine_triangulation:
      merge_threshold_image: 7
      merge_threshold_world: 0.2

# list of tasks to perform
tasks:
  - split_sequences
  - segment_pv_modules
  - track_pv_modules
  - crop_and_rectify_modules
  #- filter_out_sun_reflections  # not yet implemented, leave commented out
  - prepare_opensfm
  - opensfm_extract_metadata
  - opensfm_detect_features
  - opensfm_match_features
  - opensfm_create_tracks
  - opensfm_reconstruct
  - triangulate_pv_modules
  - refine_triangulation
  - reorganize_patches
```

#### Step 4: Process videos

Once the config file is created, you can process the data by executing the following command inside the interactive session in the Docker container
```
python main.py testing/configs/config_plant_A.yml
```

To control which tasks are executed you can (un)comment tasks under 'tasks' in the config file. Note, that you cannot skip any of the tasks, i.e. you will have to run each tasks at least once in the order specified in the config file.

We recommend to first uncomment the steps "split_sequences", "segment_pv_modules", "track_pv_modules", "crop_and_rectify_modules" and commenting all subsequent steps. These are preprocessing steps. You should ensure the correctness of the output of these steps before continuing with the remaining processing steps. To continue, comment out the first four steps and uncomment the remaining steps. Rerun `python main.py testing/configs/config_plant_A.yml`.


#### Step 5: Visualize results

We provide a script `extractor/mapping/plot_reconstruction.py` for plotting the reconstructed camera poses, PV modules and map points. You can use this script to validate whether your PV plant was reconstructed and georeferenced correctly.

To this end, run the script from within the interactive Docker session and provide the `work_dir` of the plant
```
python plot_reconstruction.py "/storage-2/pvextractor-georeferencing/Plant_A/workdir"
```
You may inspect additional optional arguments by running 
```
python plot_reconstruction.py -h
```


## Output directory structure

After completing a processing task a new subdirectory with results of that task will be created under the `work_dir` specified in the config file. E.g. after running the `split_sequences` tasks, there will be a new directory called `splitted`. Whenever you rerun a task, the respective subdirectory is deleted automatically before that task starts.

Note, that the directory tree below only shows important files and subdirectories. We will briefly explain them below.

```
/workdir
  |-- splitted
  |    |-- timestamps.csv
  |    |-- gps
  |    |     |-- gps.csv
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
```

#### splitted (split_sequences)

Contains the individual IR video frames as 16-bit TIFF images and additional 8-bit JPEG preview images. The `gps.csv` and `gps.json` files contain the longitude, latitude, and altitude (in this order) of each video frame. The `timestamps.csv` contains timestamps of each video frame.

#### segmented (segment_pv_modules)

Contains results of the Mask R-CNN instance segmentation model applied to each video frame. Specifically, this directory contains binary segmentation masks of PV modules, bounding boxes and preview images. To validate correctness of the instance segmentation you can look at the preview video in `preview.avi`.

#### tracking (track_pv_modules)

Contains the results of PV module tracking over subsequent video frames. The `tracks.csv` contains the frame name, mask name, tracking ID, and module center point in the image (x, y in pixels). Each PV module has a unique tracking ID that stays constant over subsequent video frame, in which the module is visible. To validate correctness of the module tracking you can look at the preview video in `tracks_preview.avi`.

#### patches (crop_and_rectify_modules)

Contains the cropped and rectified image patches of each PV module. The `preview` directory contains 8-bit JPEG preview images and the `radiometric` directory the respective 16-bit TIFF images. For each PV module there is a directory named after the module's tracking ID, which contains the individual image patches showing the same module in subsequent video frames. The `meta.pkl` is a Python pickle file, containing additional information about each image patch, such as the image coordinates of the module's center point, the bounding quadrilateral that was fit to the module's segmentation mask, and the homography used to rectify the module image.

#### mapping (prepare_opensfm, ..., refine_triangulation)

Contains the inputs and outputs of the tasks relating to the georeferencing of PV modules, which is performed using [OpenSfM](https://github.com/mapillary/OpenSfM). For each cluster configured in the config file there is a subdirectory, which contains the OpenSfM dataset for that cluster and which has the structure described [here](https://opensfm.org/docs/dataset.html). Most notably, each of these subdirectories contains the `reconstruction.json` file with the reconstructed camera poses and 3D map points produced by OpenSfM.<br>
The main result stored in the mapping directory is a GeoJSON file `module_geolocations_refined.geojson`. This file follows the [GeoJSON specification](https://datatracker.ietf.org/doc/html/rfc7946) and contains a feature collection of polygons and points for each PV module. The polygon resembles the longitude, latitude and altitude of the four corner points of the PV module in [WGS 84](https://epsg.io/4326) coordinates. The point is the geocoordinate of the module's center point. Each polygon and each point have a `track_id` property, which is the tracking ID of the respective PV module.

#### patches_final (reorganize_patches)

This directory has the same overall structure as the `patches` directory, but considers the merging of tracking IDs. Merged tracking IDs occur when the same PV module has two different tracking IDs, which can happen occasionally due to a tracking error. These duplicates are identified during the georeferencing procedure. If for example the trackings IDs `abc123` and `def456` belong to the same module, there will be only one directory in the `patches_final` directory named after the first tracking ID (`abc123`), which contains all module images of both subfolders `abc123` and `def456` from the `patches` directory.


## (Optional) Train the Mask R-CNN model

The project uses Mask R-CNN for instance segmentation of PV modules in IR video frames. It is pretrained on a large PV module dataset. However, if you encounter issues with the accuracy of the Mask R-CNN model, you may wish to fine-tune the model on your own dataset. For this, we recommend annotating data using the [Grid Annotation Tool](https://github.com/LukasBommes/Grid-Annotation-Tool). Data labelled with this tool canbe directly used for training the Mask R-CNN model.

To train/fine-tune Mask R-CNN start jupyter lab in the interactive Docker session
```
jupyter lab --allow-root --ip=0.0.0.0 --port=8888
```
and open the displayed URL (e.g. `http://127.0.0.1:8888/?token=4127acb479920bf6f38cb4877136b6681a50177cad9e18ab`) in the web browser on the host machine. In jupyter lab navigate to `extractor/segmentation` and start the `train.ipynb` notebook.

Prior to training the model with this script, you may need to edit the training config in `extractor/segmentation/configs.py` in the same directory. Also make sure the training dataset is available at the location specified in `DATASET_PATH` in `extractor/segmentation/configs.py`.

The training dataset should contain the following two folders:
- `images_radiometric`: 16-bit radiometric images taken from the splitted IR video file (e.g. \*.SEQ file)
- `annotations`: Corresponding \*.json file with annotations for each image. It can be created with the [Grid Annotation Tool](https://github.com/LukasBommes/Grid-Annotation-Tool).


## Citation

This repository implements our research presented in the following two papers. If you use PV-Mapper in your own research please cite these works
```
@article{Bommes.2021,
  author  = {Bommes, Lukas and Pickel, Tobias and Buerhop-Lutz, Claudia and Hauch, Jens and Brabec, Christoph and Peters, Ian Marius},
  title   = {Computer vision tool for detection, mapping, and fault classification of photovoltaics modules in aerial {IR} videos},
  journal = {Progress in Photovoltaics: Research and Applications},
  volume={29},
  number={12},
  pages={1236--1251},
  year = {2021}}
```

```
[second paper has not been not published yet]
```
