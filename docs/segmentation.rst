===========================
Segmentation using a UNet2D
===========================

0. What is UNet2D?
==================

- UNet2D is a deep-learning architecture in the family of convolutional neural-networks and in the sub-family of auto-encoders.
- It is trained through supervized learning, which means that for training, some pairs of input image + expected segmentation (== ground-truth) are required.
- After training, the model is able to produce a probability map through its process of inference. This probability map has to be thresholded to transform it into a mask.
- UNet2D generates a semantic segmentation instead of an instances segmentation. It means that each pixel will contain the answer to the question "is this pixel part of a microglia?" but the cells won't be given individual IDs.

1. Get your data ready
======================

- You can retrain the model if you have some annotated data using the provided file: src/dl/unet2d_training.py
- To train new UNet models, you need the file "src/dl/unet2d_training.py". It contains the entire workflow to produce a bundled model ready for deployment.
- Before starting, create a folder named "models" to store all the new model versions you create.
- You also need a "working_dir" where the script will export its temporary data.
- To train the UNet model, you need two distinct folders. You can name them as you like.
    - The first folder, referred to as "inputs", will contain ".tif" images with values globally normalized in the range [0.0, 1.0].
    - The second folder, referred to as "masks", will also contain ".tif" images, but these will be binary masks. They are thresholded to everything above 0 upon opening, so there is no restriction on whether they should be 0 and 1 or 0 and 255.
    - Images in both folders should be named the same way.
- The models produced by this script include:
    - "version.txt": The version index of this model, allowing detection if the model should be re-downloaded from the internet.
    - "training_history.png": A set of 2 plots (with 4 plot slots).
        - The first plot contains the loss and the validation loss.
        - The second plot contains the training and validation precision, as well as the training and validation recall.
    - "last.keras": The weights generated at the last training epoch.
    - "best.keras": The weights that achieved the best validation loss. These weights are used in the segmentation step.
    - "data_usage.json": Information on which files were used for training and which were used for validation.
    - "augmentations_preview.png": A sample of data after passing through the data augmentation pipeline.
    - "architecture.png": A graph representing the UNet2D architecture used by this model.
    - "predictions": A folder containing the validation data and the model's predictions.

2. Data augmentation
====================

The images on which we have to work have a wide variance, so we need a solid data augmentation workflow to avoid having to annotate an unreasonable number of images, and to ensure that the model generalizes well to different types of data.

The data augmentation pipeline includes the following transformations:

- **Random rotations**: The images are randomly rotated in a range from -90 to 90 degrees.
- **Random flips**: The images are randomly flipped horizontally and/or vertically.
- **Random gamma adjustment**: A random gamma correction is applied to every patch. It allows to spread or dilate the histogram.
- **Random noise addition**: Random Gaussian noise is added to the images.
- **Filament ruptures**: On many images, the filaments pass on another Z plane before coming back. To simulate that, some filamentous areas are randomly discarded (blurred).

These augmentations are applied on-the-fly at loading to ensure that each epoch sees a different set of augmented images, which helps in improving the robustness and generalization of the model.

3. Filaments extraction
=======================

- Images containing filaments actually contain 96% of background, which represents a massive imbalance between the background and foreground classes.
- Using a usual loss function would end-up in the model predicting only solid black patches, as it would consider that it is 96% correct.
- To address that problem, we had to re-implement the clDice loss as described in the paper: https://doi.org/10.48550/arXiv.2404.00130.

4. Setup
========

- If you already have a Python environment in which "Microglia Analyzer" is installed, it already contains everything you need to train a model.
- To launch the training, you just have to fill the settings in the first section, and run the script.

+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| Name                 | Description                                                                                                                    |
+======================+================================================================================================================================+
| data_folder          | Parent folder of the "inputs" and "masks" folders.                                                                             |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| qc_folder            | Parent folder of the "inputs" and "masks" folders used only for quality control (not for training). These are just images used |
|                      | to perform performance metrics at the end of training.                                                                         |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| inputs_name          | Name of the folder containing the input images (name of the folder in `data_folder` and `qc_folder`).                          |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| masks_name           | Name of the folder containing the masks (name of the folder in `data_folder` and `qc_folder`).                                 |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| models_path          | Folder in which the models will be saved. They will be saved as "{model_name_prefix}-V{version_number}".                       |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| working_directory    | Folder in which the training, validation, and testing folders will be created. This folder and its content can be deleted once |
|                      | the training is done.                                                                                                          |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| model_name_prefix    | Prefix of the model name. Will be part of the folder name in `models_path`.                                                    |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| reset_local_data     | If True, the locally copied training, validation, and testing folders will be re-imported.                                     |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| remove_wrong_data    | If True, the data that is not useful will be deleted from the data folder. It is a destructive operation, review the first run |
|                      | of the sanity check before activating this.                                                                                    |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| data_usage           | Path to a JSON file containing how each input file should be used (for training, validation, or testing).                      |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| validation_percentage| Percentage of the data that will be used for validation. This data will be moved to the validation folder.                     |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| batch_size           | Number of images per batch.                                                                                                    |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| epochs               | Number of epochs for the training.                                                                                             |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| unet_depth           | Depth of the UNet model, i.e., the number of layers in the encoder part (equal to the number of layers in the decoder part).   |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| num_filters_start    | Number of filters in the first layer of the UNet.                                                                              |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| dropout_rate         | Dropout rate. Percentage of neurons that will be randomly disabled at each epoch. Better for generalization.                   |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| optimizer            | Optimizer used for the training.                                                                                               |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| learning_rate        | Learning rate at which the optimizer is initialized.                                                                           |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| skeleton_coef        | Coefficient of the skeleton loss.                                                                                              |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| bce_coef             | Coefficient of the binary cross-entropy loss.                                                                                  |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| early_stop_patience  | Number of epochs without improvement before stopping the training.                                                             |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| dilation_kernel      | Kernel used for the dilation of the skeleton.                                                                                  |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| loss                 | Loss function used for the training.                                                                                           |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| use_data_augmentation| If True, data augmentation will be used.                                                                                       |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| use_mirroring        | If True, random mirroring will be used.                                                                                        |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| use_gaussian_noise   | If True, random Gaussian noise will be used.                                                                                   |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| noise_scale          | Scale of the Gaussian noise (range of values).                                                                                 |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| use_random_rotations | If True, random rotations will be used.                                                                                        |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| angle_range          | Range of the random rotations. The angle will be in [angle_range[0], angle_range[1]].                                          |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| use_gamma_correction | If True, random gamma correction will be used.                                                                                 |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| gamma_range          | Range of the gamma correction. The gamma will be in [1 - gamma_range, 1 + gamma_range] (1.0 == neutral).                       |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| use_holes            | If True, holes will be created in the input images to teach the network to fill them.                                          |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+
| export_aug_sample    | If True, an augmented sample will be exported to the working directory as a preview.                                           |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------+

5. Usage
========

- This model consumes patches of 512×512 pixels, with an overlap of 128 pixels.
- The merging is performed with the alpha-blending technique described on the page where the patches creation is explained.
- The output is labeled by connected components and filtered by number of pixels (processed from a minimal area in µm²) before being presented to the user.
