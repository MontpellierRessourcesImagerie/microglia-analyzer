=============================
Classification using a YOLOv5
=============================

1. Get your data ready
======================

- To train new YOLO models, you need the file :code:`src/dl/yolov5_training.py`. It contains the entire workflow to produce a bundled model ready for deployment.
- Before starting, create a folder named :code:`models` to store all the new model versions you create.
- You also need a :code:`working_dir` where the script will export its temporary data.
- To train the YOLO model, you need two distinct folders. You can name them as you like.
    - The first folder, referred to as :code:`images`, will contain :code:`.png` images with values globally normalized in the range [0, 255].
    - The second folder, referred to as :code:`labels`, will also contain :code:`.txt` files in which each line contains: C, X1, Y1, X2, Y2.
    - The names in both folders should be the same with only the exntension changing.
- The models produced by this script include:
    - :code:`version.txt`: The version index of this model, allowing detection if the model should be re-downloaded from the internet.
    - :code:`val_batch0_pred.jpg`: An overview of what the model predicted on an image from the validation set.
    - :code:`val_batch0_labels.jpg`: The ground-truth of the image above.
    - :code:`train_batchN.jpg`: A sample of N images and their labels (ground-truth) from the training-set.
    - :code:`results.png`: Plot of metrics along the training (box loss, object loss, class loss, precesion & recall).
    - :code:`results.csv`: Actual values that were plotted in :code:`results.png`.
    - :code:`R_curve.png`: Recall over epochs plotted.
    - :code:`P_curve.png`: Precision over epochs plotted.
    - :code:`PR_curve.png`: Precision against recall plotted.
    - :code:`F1_curve.png`: F1 score over epochs.
    - :code:`opt.yml`: Settings used to traing the models.
    - :code:`labels_correlogram.jpg`: Distribution of the bounding-boxes locations and dimensions.
    - :code:`labels.jpg`: Distribution of the number of labels for each class, and their location in images.
    - :code:`hyp.yaml`: Hyper-parameters used to train the model.
    - :code:`confusion_matrix.png`: Confusion matrix (count per class of what was predicted vs. what was expected) of the created model on the validation set.
    - :code:`weights`: A folder containing :code:`best.pt` and :code:`last.pt` which are the actual trained models.

2. Data augmentation
====================

- **HSV-Hue Augmentation**: The hue augmentation factor for HSV color space. Here, we work on grayscale image, so the provided value doesn't matter.
- **HSV-Saturation Augmentation**: The saturation augmentation factor for HSV color space.
- **HSV-Value Augmentation**: The value augmentation factor for HSV color space. It was blocked to 0.01 to avoid making objets in the background visible.
- **Rotation Degrees**: The maximum rotation degrees for data augmentation. Here, we allowed a range of 90° in either directions.
- **Translation**: The maximum translation factor for data augmentation. Our objects can be anywhere on images, so we allowed a range of half the image size of each axis.
- **Scale**: The scaling factor for data augmentation. The scale matters a lot to classify microglia, so it was locked to 1.0.
- **Vertical Flip Probability**: The probability of performing a vertical flip during data augmentation.
- **Horizontal Flip Probability**: The probability of performing a horizontal flip during data augmentation.
- **Mosaic Augmentation**: The factor for mosaic data augmentation. To get more data, we create mosaics of inputs images to create :code:`new images`.

3. Setup
========

- Required to use YOLOv5m because there was not enough learning capacity avec YOLOv5s

+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
+ Settings              | Description                                                                                                                                           |
+=======================+=======================================================================================================================================================+
| data_folder           | Parent folder of the :code:`images` and :code:`labels` folders.                                                                                       |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| qc_folder             | Parent folder of the quality-control images (also :code:`images` and :code:`labels` folders).                                                         |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| inputs_name           | Name of the folder containing the inputs images and the QC inputs.                                                                                    |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| annotations_name      | Name of the folder containing the labels for the training and QC.                                                                                     |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| models_path           | Root of the folder in which models will be stored.                                                                                                    |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| working_directory     | Directory in which the scripts creates its temporary data. Can be deleted after training.                                                             |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| model_name_prefix     | Prefix that will be given to the folders containing newly created models.                                                                             |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| reset_local_data      | Should the local set of images (== the data in the working directory) be reseted at every training. Recommended.                                      |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| validation_percentage | Percentage of the provided data that will be used for the validation step.                                                                            |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| batch_size            | Number of images processed at the same time while training. Should be as high as your memory can handle.                                              |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| epochs                | Number of times that the whole data will be seen during training.                                                                                     |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| classes_names         | List of class names that should be predicted by the model being trained.                                                                              |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| optimizer             | Optimizer used for the gradient descent.                                                                                                              |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| deterministic         | Should the inference be deterministic (one input always give the same output). Works by using a random seed if False.                                 |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| cos_lr                | Usually, the learning rate dicreases as the epochs go. If True, it will rather follow a sinusoidal curve, starting on a maxima (hence the cosine)     |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| label_smoothing       | Should the probability map of classes be smoothed (blurred) before building bounding boxes.                                                           |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| dropout               | Percentage of neurons randomly disabled at each epoch to improve the generalization.                                                                  |
+-----------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+

4. Bind classification to segmentation
======================================

- By the end of the classification process, we have a set of bounding-boxes deduced by the model. Each box has a class (garbage, amoeboid, intermediate or homeostatic).
- At the previous step, we built masks representing microglia.
- However, at this point, there is no relation between the segmentation and the classification. We need to bind each item from the mask to a class.
- To do that, a system of vote was implemented.
    - Each object starts with a set of N bins, with N being the number of classes. Along the process, each bin will count the number of votes for each class.
    - For each bounding-box predicted by YOLO, we search the biggest object inside it, and designate it as the target.
    - In the target, the bin corresponding to the bounding-box's class will receive P×S votes with:
        - P: The number of the target's pixels in the bounding-box.
        - S: The certainty score of the bounding-box.
    - At the end, we take the major vote for each object.
    - An object with no vote is automatically declared garbage.