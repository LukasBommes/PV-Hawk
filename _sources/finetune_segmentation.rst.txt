Finetuning the Mask R-CNN Model
===============================

PV Hawk uses Mask R-CNN for instance segmentation of PV modules in IR video frames. It is pretrained on a large PV module dataset. However, if you encounter issues with the accuracy of the Mask R-CNN model, you may wish to fine-tune the model on your own dataset. For this, we recommend annotating data using the `Grid Annotation Tool <https://github.com/LukasBommes/Grid-Annotation-Tool>`_. Data labelled with this tool can be directly used for training the Mask R-CNN model.

To train/fine-tune Mask R-CNN run the Docker container as described :ref:`here <run-the-docker-image>` and start jupyter lab inside the Docker shell with

.. code-block:: console

  jupyter lab --allow-root --ip=0.0.0.0 --port=8888
 
Note, that the port number must match the port forwarded when starting the Docker container. Depending on the location of your training dataset, you may have to add another mapping statement (`-v ...`) to the Docker run command to map your training dataset into the container.

Open the displayed URL in the web browser on your machine. In jupyter lab navigate to `extractor/segmentation` and open the `train.ipynb` notebook.

Prior to training the model with this script, you need to edit the `DATASET_TRAIN_PATH`, `DATASET_VAL_PATH` in the training config in `extractor/segmentation/configs.py`. The paths refer to the location in the Docker container to which you mapped your training dataset when starting the Docker container. You may also have to change the `MEAN_PIXEL` values.

The training dataset should contain the following two folders:

- `images_radiometric`: Your custom 16-bit IR video frames containing PV modules.
- `annotations`: JSON file for each image with annotated PV modules. The annotation can be created with the `Grid Annotation Tool <https://github.com/LukasBommes/Grid-Annotation-Tool>`_.

If you want to start training from MS COCO pretrained weights, you can download the corresponding weights file from `here <https://drive.google.com/file/d/1x-Q79OxMqoFaXLh6IguB1UCV4RZys49J/view?usp=sharing>`_ and move it to `extractor/segmentation/Mask_RCNN`.

After training weights of the new model will be available in `extractor/segmentation/Mask_RCNN/ logs/pv_modules\<timestamp\>/mask_rcnn_pv_modules_\<epoch\>.h5` where `<timestamp>` is the timestamp at the beginning of training and `<epoch>` the number of epochs the model was trained. To use the newly trained model set the `WEIGHTS_FILE` parameter in `extractor/segmentation/configs.py` to the path of the `*.h5` weights file.
