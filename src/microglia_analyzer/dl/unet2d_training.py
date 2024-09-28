import tifffile
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import re

from scipy.ndimage import rotate
import pandas as pd
from tabulate import tabulate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, TensorBoard, Callback)
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     UpSampling2D, concatenate, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              SETTINGS                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 1. SETTINGS

"""

- `data_folder`      : Folder in which we can find the images and masks folders.
- `qc_folder`        : Folder in which we can find the quality control images and masks folders.
- `inputs_name`      : Name of the folder containing the input images (name of the folder in `data_folder` and `qc_folder`.).
- `masks_name`       : Name of the folder containing the masks (name of the folder in `data_folder` and `qc_folder`.).
- `models_path`      : Folder in which the models will be saved. They will be saved as "{model_name_prefix}-V{version_number}".
- `working_directory`: Folder in which the training, validation and testing folders will be created.
- `model_name_prefix`: Prefix of the model name. Will be part of the folder name in `models_path`.
- `reset_local_data` : If True, the locally copied training, validation and testing folders will be re-imported.

- `validation_percentage`: Percentage of the data that will be used for validation. This data will be moved to the validation folder.
- `batch_size`           : Number of images per batch.
- `epochs`               : Number of epochs for the training.
- `unet_depth`           : Depth of the UNet model == number of layers in the encoder part (== number of layers in the decoder part).
- `num_filters_start`    : Number of filters in the first layer of the UNet.
- `dropout_rate`         : Dropout rate.
- `optimizer`            : Optimizer used for the training.
- `learning_rate`        : Learning rate at which the optimizer is initialized

- `use_data_augmentation`: If True, data augmentation will be used.
- `use_mirroring`        : If True, random mirroring will be used.
- `use_gaussian_noise`   : If True, random gaussian noise will be used.
- `use_random_rotations` : If True, random rotation of 90, 180 or 270 degrees will be used.
- `use_gamma_correction` : If True, random gamma correction will be used.
- `gamma_range`          : Range of the gamma correction. The gamma will be in [1 - gamma_range, 1 + gamma_range] (1.0 == neutral).

"""

#@markdown ## 📍 a. Data paths

data_folder       = "/home/benedetti/Documents/projects/2060-microglia/data/unet-training/"
qc_folder         = None
inputs_name       = "images"
masks_name        = "masks"
models_path       = "/home/benedetti/Documents/projects/2060-microglia/data/unet-training/models/"
working_directory = "/tmp/unet_working/"
model_name_prefix = "µnet"
reset_local_data  = True
remove_wrong_data = True

#@markdown ## 📍 b. Network architecture

validation_percentage = 0.15
batch_size            = 32
epochs                = 50
unet_depth            = 4
num_filters_start     = 16
dropout_rate          = 0.25
optimizer             = 'Adam'
learning_rate         = 0.0001

#@markdown ## 📍 c. Data augmentation

use_data_augmentation = True
use_mirroring         = True
use_gaussian_noise    = True
use_random_rotations  = True
angle_range           = (-30, 30)
use_gamma_correction  = True
gamma_range           = (0.2, 4.0)
show_preview          = False


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            SANITY CHECK                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 2. SANITY CHECK

"""
The goal of this section is to make sure that the data located in the `data_folder` is consistent.
The following checks will be performed:
    - All files must be TIFF images ('.tif' or '.tiff', whatever the case).
    - Each file must be present in all the folders (images and masks).
    - The shape (X, Y and Z dimensions in pixels) of the images must be the same.
    - All the data must be useful, it implies that:
        | Input images have more than 1e-6 between the maximum and minimum values.
        | Masks must be binary masks (on 8-bits with only 0 and another value).
"""

#@markdown ## 📍 a. Data check

# Regex matching a TIFF file, whatever the case and the number of 'f'.
_TIFF_REGEX = r".+\.tiff?"

def get_data_pools(root_folder, folders, tif_only=False):
    """
    Aims to return the files available for training in every folder (not path).
    Probes the content of the data folders provided by the user.
    Both the images and the masks are probed.
    It is possible to filter the files to keep only the tiff files, whatever the case (Hi Windows users o/).
    In the returned tuple, the first element is a list (not a dict) following the same order as the 'folders' list.

    Args:
        root_folder (str): The root folder containing the images and masks folders.
        folders (list): The list of folders to probe.
        tif_only (bool): If True, only the tiff files will be kept.
    
    Returns:
        tuple: (pool of files per individual folder, the set of all the files found everywhere merged together.)
    """
    pools = [] # Pools of files found in the folders.
    all_data = set() # All the names of files found gathered together.
    for f in folders: # Fetching content from folders
        path = os.path.join(root_folder, f)
        pool = set([i for i in os.listdir(path)])
        if tif_only:
            pool = set([i for i in pool if re.match(_TIFF_REGEX, i, re.IGNORECASE)])
        pools.append(pool)
        all_data = all_data.union(pool)
    return pools, all_data

def get_shape():
    """
    Searches for the first image in the images folder to determine the input shape of the model.

    Returns:
        tuple: The shape of the input image.
    """
    _, l_files = get_data_pools(data_folder, [inputs_name], True)
    input_path = os.path.join(data_folder, inputs_name, list(l_files)[0])
    raw = tifffile.imread(input_path)
    s = raw.shape
    if len(s) == 2:
        s = (s[0], s[1], 1)
    return s

def is_extension_correct(root_folder, folders):
    """
    Checks that the files are all TIFF images.

    Args:
        root_folder (str): The root folder containing the images and masks folders
        folders (list): The list of folders to probe (these folders must be in `root_folder`).

    Returns:
        dict: Keys are files, values are booleans. True if the file is a TIFF image, False otherwise.
    """
    _, all_data = get_data_pools(root_folder, folders)
    _, all_tiff = get_data_pools(root_folder, folders, True)
    extensions = {k: (k in all_tiff) for k in all_data}
    return extensions

def is_data_shape_identical(root_folder, folders):
    """
    All the data must be the same shape in X, Y and Z.

    Args:
        root_folder (str): The root folder containing the images and masks folders.
        folders (str): The list of folders to probe (these folders must be in `root_folder`).

    Returns:
        dict: Keys are files, values are booleans. True if the shape is identical, False otherwise.
    """
    _, all_data = get_data_pools(root_folder, folders, True)
    ref_size = None
    shapes = {k: False for k in all_data}
    for file in all_data:
        for folder in folders:
            path = os.path.join(root_folder, folder, file)
            if not os.path.isfile(path):
                continue
            img_data = tifffile.imread(path)
            if ref_size is None:
                ref_size = img_data.shape
            if img_data.shape == ref_size:
                shapes[file] = True
    return shapes

def is_data_useful(root_folder, folders):
    """
    There must not be empty masks or empty images.

    Args:
        root_folder (str): The root folder containing the images and masks folders

    Returns:
        bool: True if the data is consistent, False otherwise.
    """
    images_path = os.path.join(root_folder, inputs_name)
    masks_path = os.path.join(root_folder, masks_name)
    _, all_data = get_data_pools(root_folder, folders, True)
    useful_data = {k: False for k in all_data}

    for file in all_data:
        img_path = os.path.join(images_path, file)
        mask_path = os.path.join(masks_path, file)
        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            continue
        img_data = tifffile.imread(img_path)
        mask_data = tifffile.imread(mask_path)
        s = True
        if np.nan in set(np.unique(mask_data)).union(set(np.unique(img_data))):
            s = False
        if np.max(img_data) - np.min(img_data) < 1e-6:
            s = False
        if len(np.unique(mask_data)) < 2: # Want binary mask or labels
            s = False
        useful_data[file] = s
    return useful_data

def is_matching_data(root_folder, folders):
    """
    Every file must be present in every folder.
    Lists every possible file and verifies that it's present everywhere.

    Args:
        root_folder (str): The root folder containing the images and masks folders
        folders (list): The list of folders to probe (these folders must be in `root_folder`).

    Returns:
        dict: Keys are files, values are booleans. True if the file is present everywhere, False otherwise.
    """
    pools, all_data = get_data_pools(root_folder, folders)
    matching_data   = {k: False for k in all_data}
    for data in all_data:
        status = [False for _ in range(len(folders))]
        for i, pool in enumerate(pools):
            if data in pool:
                status[i] = True
        matching_data[data] = all(status)
    return matching_data

def merge_dicts(d1, d2):
    """
    Transfers the values of d2 to d1 if and only if the key doesn't exist in d1.
    Keys present in d1 are not edited with the value they have in d2.
    """
    for key, value in d2.items():
        if key not in d1:
            d1[key] = value


#@markdown ## 📍 b. Sanity check launcher

_SANITY_CHECK = [
    ("extension", is_extension_correct),
    ("pair"     , is_matching_data),
    ("useful"   , is_data_useful),
    ("shape"    , is_data_shape_identical)
]

_RESET      = "\033[0m"
_GREEN      = "\033[32m"
_RED_BOLD   = "\033[1;31m"
_INSANITIES = {
    "extension": (f"{_GREEN}OK{_RESET}", f"{_RED_BOLD}UNKNOWN{_RESET}"),
    "pair"     : (f"{_GREEN}OK{_RESET}", f"{_RED_BOLD}MISSING{_RESET}"),
    "useful"   : (f"{_GREEN}OK{_RESET}", f"{_RED_BOLD}USELESS{_RESET}"),
    "shape"    : (f"{_GREEN}OK{_RESET}", f"{_RED_BOLD}MISMATCH{_RESET}")
}

def apply_verbose(results):
    verbose = {k: v.copy() for k, v in results.items()}
    for test, pool in results.items():
        for file, status in pool.items():
            if not status:
                verbose[test][file] = _INSANITIES[test][1]
            else:
                verbose[test][file] = _INSANITIES[test][0]
    return verbose

def sanity_check(root_folder):
    folders = [inputs_name, masks_name]
    results = {}
    _, all_data = get_data_pools(root_folder, folders)
    false_data = {k: False for k in all_data}
    for name, func in _SANITY_CHECK:
        results[name] = func(root_folder, folders)
        merge_dicts(results[name], false_data)
    assessment = [all(v.values()) for v in results.values()]
    verbose = apply_verbose(results)
    df = pd.DataFrame(verbose)
    df = df.sort_index()
    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
    return (all(assessment), results)


#@markdown ## 📍 c. Remove dirty data

def remove_dirty_data(root_folder, folders, results):
    """
    Removes the files that are not useful.
    """
    trash_path = os.path.join(working_directory, "trash")
    for f in folders:
        os.makedirs(os.path.join(trash_path, f), exist_ok=True)
    for test, pool in results.items():
        for file, status in pool.items():
            if not status:
                for f in folders:
                    path = os.path.join(root_folder, f, file)
                    if os.path.isfile(path):
                        shutil.move(path, os.path.join(trash_path, f, file))
    print(f"🗑️  Dirty data has been moved to: {trash_path}.")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            DATA MIGRATION                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 3. DATA MIGRATION

#@markdown ## 📍 a. Utils

_LOCAL_FOLDERS = ["training", "validation", "testing"]

def create_local_dirs(reset=False):
    """
    This function is useless if you don't run the code on Google Colab, or any other cloud service.
    Basically, data access is way faster if you copy the data to the local disk rather than a distant server.
    Since the data is accessed multiple times during the training, the choice was made to migrate the data to the local disk.
    There is a possibility to reset the data, in case you want to shuffle your data for the next training.

    Args:
        reset (bool): If True, the folders will be reset.
    """
    if not os.path.isdir(working_directory):
        raise ValueError(f"Working directory '{working_directory}' does not exist.")
    leaves = [inputs_name, masks_name]
    for r in _LOCAL_FOLDERS:
        for l in leaves:
            path = os.path.join(working_directory, r, l)
            if os.path.isdir(path) and reset:
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

def copy_to(src_folder, dst_folder, files):
    """
    Copies a list of files from a source folder to a destination folder.

    Args:
        src_folder (str): The source folder.
        dst_folder (str): The destination folder.
        files (list): The list of files to copy.
    """
    for f in files:
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(dst_folder, f)
        shutil.copy(src_path, dst_path)

def check_sum(targets):
    """
    Since we move some fractions of data to some other folders, we need to check that the sum of the ratios is equal to 1.
    Otherwise, we would have some data missing or we would try to read data that doesn't exist.
    """
    acc = sum([i[1] for i in targets])
    return abs(acc - 1.0) < 1e-6

def migrate_data(targets, source):
    """
    Copies the content of the source folder to the working directory.
    The percentage of the data to move is defined in the targets list.
    Meant to work with pairs of files.

    Args:
        targets (list): List of tuples. The first element is the name of the folder, the second is the ratio of the data to move.
        source (str): The source folder
    """
    if not check_sum(targets):
        raise ValueError("The sum of the ratios must be equal to 1.")
    folders = [inputs_name, masks_name]
    _, all_data = get_data_pools(source, folders, True)
    all_data = list(all_data)
    random.shuffle(all_data)
    last = 0
    for target, ratio in targets:
        n = int(len(all_data) * ratio)
        copy_to(os.path.join(source, inputs_name), os.path.join(working_directory, target, inputs_name), all_data[last:last+n])
        copy_to(os.path.join(source, masks_name), os.path.join(working_directory, target, masks_name), all_data[last:last+n])
        last += n


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            DATA AUGMENTATION                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 4. DATA AUGMENTATION

#@markdown ## 📍 a. Data augmentation functions

def random_flip(image, mask):
    """
    Applies a random horizontal or vertical flip to both the image and the mask.
    
    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The input mask.
        
    Returns:
        (np.ndarray, np.ndarray): The flipped image and mask.
    """
    # Horizontal flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    # Vertical flip
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    
    return image, mask

def random_rotation(image, mask):
    """
    Applies a random rotation (by any angle) to both the image and the mask.
    The image uses bilinear interpolation, while the mask uses nearest-neighbor interpolation to avoid grayscale artifacts.
    
    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The input mask.
        angle_range (tuple): The range of angles (in degrees) from which to sample the random rotation angle.
        
    Returns:
        (np.ndarray, np.ndarray): The rotated image and mask.
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    rotated_image = rotate(image, angle, reshape=False, order=1, mode='reflect')
    rotated_mask = rotate(mask, angle, reshape=False, order=0, mode='reflect')
    
    return rotated_image, rotated_mask

def gamma_correction(image, mask):
    """
    Applies a random gamma correction to the image. The mask remains unchanged.
    
    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The input mask.
        gamma_range (tuple): The range from which to sample the gamma value.
        
    Returns:
        (np.ndarray, np.ndarray): The gamma-corrected image and the unchanged mask.
    """
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    image = np.power(image, gamma)
    
    return image, mask

def normalize(image, mask):
    """
    Normalizes the image values to be between 0 and 1. The mask remains unchanged.
    
    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The input mask.
        
    Returns:
        (np.ndarray, np.ndarray): The normalized image and the unchanged mask.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    normalized_image = (image - min_val) / (max_val - min_val)
    
    return normalized_image, mask

def apply_data_augmentation(image, mask):
    """
    Applies all the data augmentation functions to the image and the mask.

    Args:
        image (tf.Tensor): The input image.
        mask (tf.Tensor): The input mask.
    
    Returns:
        (tf.Tensor, tf.Tensor): The augmented image and mask.
    """
    if use_mirroring:
        image, mask = random_flip(image, mask)
    if use_random_rotations:
        image, mask = random_rotation(image, mask)
    if use_gamma_correction:
        image, mask = gamma_correction(image, mask)
    image, mask = normalize(image, mask)
    return image, mask

#@markdown ## 📍 b. Datasets visualization

def visualize_augmentations(num_examples=5):
    """
    Visualizes original and augmented images side by side.
    
    Args:
        num_examples (int): The number of examples to visualize.
    """
    s = get_shape()
    ds = make_dataset("training", True).batch(1).take(num_examples)

    grid_size = (2, num_examples)
    fig, axes = plt.subplots(*grid_size, figsize=(15, 6))

    for i, (augmented_image, original_image) in enumerate(ds):
        if i >= num_examples:
            break
        
        augmented_image = augmented_image[0].numpy()
        original_image = original_image[0].numpy()
        
        axes[0, i].imshow(original_image)
        axes[0, i].set_title(f"Mask {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(augmented_image)
        axes[1, i].set_title(f"Input {i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            DATASET GENERATOR                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 5. DATASET GENERATOR

#@markdown ## 📍 a. Datasets generator

def open_pair(input_path, mask_path, training, img_only):
    raw_img = tifffile.imread(input_path)
    raw_img = np.expand_dims(raw_img, axis=-1)
    raw_mask = (tifffile.imread(mask_path) > 0).astype(np.uint8)
    raw_mask = np.expand_dims(raw_mask, axis=-1)
    if training:
        raw_img, raw_mask = apply_data_augmentation(raw_img, raw_mask)
    image = tf.constant(raw_img, dtype=tf.float32)
    mask = tf.constant(raw_mask, dtype=tf.uint8)
    if img_only:
        return image
    else:
        return (image, mask)

def pairs_generator(src, training, img_only):
    source = src.decode('utf-8')
    _, l_files = get_data_pools(os.path.join(working_directory, source), [inputs_name], True)
    l_files = sorted(list(l_files))
    i = 0
    while i < len(l_files):
        input_path = os.path.join(working_directory, source, inputs_name, l_files[i])
        mask_path = os.path.join(working_directory, source, masks_name, l_files[i])
        yield open_pair(input_path, mask_path, training, img_only)
        i += 1

def make_dataset(source, training=False, img_only=False):
    shape = get_shape()
    
    output_signature=tf.TensorSpec(shape=shape, dtype=tf.float32, name=None)
    if not img_only:
        output_signature = (output_signature, tf.TensorSpec(shape=shape, dtype=tf.uint8, name=None))
    
    ds = tf.data.Dataset.from_generator(
        pairs_generator,
        args=(source, training, img_only),
        output_signature=output_signature
    )
    return ds

def test_ds_consumer():
    batch = 20 # will be equivalent to the batch size
    take = 10 # will be equivalent to the number of epochs
    
    ds_counter = make_dataset("training", True)
    for i, (image, mask) in enumerate(ds_counter.repeat().batch(batch).take(take)):
        print(f"{str(i+1).zfill(2)}: ", image.shape, mask.shape)

    print("\n================\n")

    ds_counter = make_dataset("training", False, True)
    for i, image in enumerate(ds_counter.repeat().batch(batch).take(take)):
        print(f"{str(i+1).zfill(2)}: ", image.shape)
    
    print("\nDONE.")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            MODEL GENERATOR                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 6. MODEL GENERATOR

#@markdown ## 📍 a. Utils

def get_version():
    """
    Used to auto-increment the version number of the model.
    Since each model is saved in a separate folder, we need to find the latest version number.
    Starts at 1 when the destination folder is empty.

    Returns:
        int: The next version number, that doesn't exist yet in the models folder.
    """
    if not os.path.isdir(models_path):
        os.makedirs(models_path)
    content = sorted([f for f in os.listdir(models_path) if f.startswith(model_name_prefix) and os.path.isdir(os.path.join(models_path, f))])
    if len(content) == 0:
        return 1
    else:
        return int(content[-1].split('-')[-1].replace('V', '')) + 1

#@markdown ## 📍 b. UNet2D architecture

def create_unet2d_model(input_shape):
    """
    Generates a UNet2D model with ReLU activations after each Conv2D layer.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # Encoder:
    skip_connections = []
    for i in range(unet_depth):
        num_filters = num_filters_start * 2**i
        x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x) 
        x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x) 
        skip_connections.append(x)
        x = MaxPooling2D(2)(x)
        x = Dropout(dropout_rate)(x)

    # Decoder:
    for i in reversed(range(unet_depth)):
        num_filters = num_filters_start * 2**i
        x = UpSampling2D(2)(x)
        x = concatenate([x, skip_connections[i]])
        x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x) 
        x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x) 
        x = Dropout(dropout_rate)(x)

    # Output layer with sigmoid for binary classification
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


#@markdown ## 📍 c. Alternative loss functions

def jaccard_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return 1 - (intersection + 1) / (union + 1)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = - alpha_t * (1 - p_t) ** gamma * tf.math.log(p_t + 1e-5)
        return tf.reduce_mean(fl)
    return focal_loss_fixed

def tversky_loss(alpha=0.5, beta=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_pos = tf.reduce_sum(y_true * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)
        return 1 - (true_pos + 1) / (true_pos + alpha * false_neg + beta * false_pos + 1)
    return loss

#@markdown ## 📍 d. Model instanciator

def instanciate_model():
    input_shape = get_shape()
    model = create_unet2d_model(input_shape)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss=bce_dice_loss, 
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Accuracy(),
            tf.keras.metrics.MeanIoU(num_classes=2)
        ]
    )
    return model


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            TRAINING THE MODEL                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 7. TRAINING THE MODEL

#@markdown ## 📍 a. Creating callback for validation

class SavePredictionsCallback(Callback):
    def __init__(self, num_examples=5):
        """
        Custom callback to save predictions to images at the end of each epoch.

        Args:
            validation_data (tf.data.Dataset): The validation dataset to predict on.
            output_dir (str): The directory where images will be saved.
            num_examples (int): The number of examples to save at each epoch.
        """
        super().__init__()
        self.validation_data = make_dataset("validation", False)
        self.output_dir = os.path.join(working_directory, "predictions")
        self.num_examples = num_examples

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_masks = next(iter(self.validation_data.batch(self.num_examples)))
        predictions = self.model.predict(val_images)

        # Create a subfolder for each epoch
        epoch_dir = os.path.join(self.output_dir, f'epoch_{epoch + 1}')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        # Save the results (input, predicted mask, and ground truth mask)
        for i in range(self.num_examples):
            tifffile.imwrite(os.path.join(epoch_dir, f'input_{str(i + 1).zfill(5)}.tif'), val_images[i].numpy())
            tifffile.imwrite(os.path.join(epoch_dir, f'mask_{str(i + 1).zfill(5)}.tif'), val_masks[i].numpy())
            tifffile.imwrite(os.path.join(epoch_dir, f'prediction_{str(i + 1).zfill(5)}.tif'), predictions[i])
            

#@markdown ## 📍 b. Training launcher

def train_model(model, train_dataset, val_dataset):
    # Path of the folder in which the model is exported.
    v = get_version()
    version_name = f"{model_name_prefix}-V{str(v).zfill(3)}"
    output_path = os.path.join(models_path, version_name)
    os.makedirs(output_path)
    plot_model(model, to_file=os.path.join(output_path, 'architecture.png'), show_shapes=True)

    print(f"💾 Exporting model to: {output_path}")

    checkpoint = ModelCheckpoint(os.path.join(output_path, 'best.keras'), save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, mode='min')

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr, SavePredictionsCallback()],
        verbose=1
    )

    model.save(os.path.join(output_path, 'last.keras'))
    return history


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            EVALUATE THE MODEL                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 8. EVALUATE THE MODEL

def plot_training_history(history):
    """
    Plots the training and validation metrics from the model's history.

    Args:
        history (History): The history object returned by model.fit().
    """
    # Retrieve metrics from history
    metrics = history.history
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Training and Validation Loss
    axes[0, 0].plot(metrics['loss'], label='Training Loss')
    axes[0, 0].plot(metrics['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Plot IoU (Jaccard Index) if available
    if 'iou' in metrics:
        axes[1, 0].plot(metrics['iou'], label='Training IoU')
        axes[1, 0].plot(metrics['val_iou'], label='Validation IoU')
        axes[1, 0].set_title('Intersection over Union (IoU)')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()

    # Plot Precision and Recall if available
    if 'precision' in metrics and 'recall' in metrics:
        axes[1, 1].plot(metrics['precision'], label='Training Precision')
        axes[1, 1].plot(metrics['val_precision'], label='Validation Precision')
        axes[1, 1].plot(metrics['recall'], label='Training Recall')
        axes[1, 1].plot(metrics['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              MAIN FUNCTION                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def main():
    from pprint import pprint

    # 1. Running the sanity checks
    data_sanity, results = sanity_check(data_folder)
    qc_sanity = True
    if qc_folder is not None:
        qc_sanity, _ = sanity_check(qc_folder)
    
    if not data_sanity:
        if remove_wrong_data:
            remove_dirty_data(data_folder, [inputs_name, masks_name], results)
        else:
            print(f"ABORT. 😱 Your {'data' if not data_sanity else 'QC data'} is not consistent. Use the content of the sanity check table above to fix all that and try again.")
            return
    else:
        print("👍 Your training data looks alright!")

    if qc_folder is None:
        print("🚨 No QC data provided.")
    elif not qc_sanity:
        print("🚨 Your QC data is not consistent. Use the content of the sanity check table above to fix all that and try again.")
    else:
        print("👍 Your QC data looks alright!")
    
    # 2. Migrate the data locally
    create_local_dirs(reset_local_data)
    migrate_data([
        ("training", 1.0-validation_percentage),
        ("validation", validation_percentage)
        ], data_folder)
    if qc_folder is not None:
        migrate_data([
            ("testing", 1.0)
            ], qc_folder)
    
    # 3. Preview the effects of data augmentation
    if show_preview:
        visualize_augmentations()

    # 4. Creating the model
    model = instanciate_model()
    model.summary()

    # 5. Create the datasets
    training_dataset   = make_dataset("training", True).repeat().batch(batch_size).take(batch_size)
    validation_dataset = make_dataset("validation", False).repeat().batch(16).take(16)
    print(f"   • Training dataset: {len(list(training_dataset))} ({training_dataset}).")
    print(f"   • Validation dataset: {len(list(validation_dataset))} ({validation_dataset}).")
    
    testing_dataset = None
    if qc_folder is not None:
        testing_dataset = make_dataset("testing", False).repeat().batch(batch_size).take(batch_size)
        print(f"   • Testing dataset: {len(list(testing_dataset))} ({testing_dataset}).")
    else:
        print("   • No testing dataset provided.")
    
    # 6. Training the model
    history = train_model(model, training_dataset, validation_dataset)

if __name__ == "__main__":
    main()