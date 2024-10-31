import tifffile
import numpy as np
import os
import shutil

from microglia_analyzer.tiles.recalibrate import recalibrate_image
from microglia_analyzer.tiles.tiler import ImageTiler2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model

_PATCH_SIZE = 512
_OVERLAP = 128

class MicrogliaAnalyzer(object):
    
    def __init__(self, logging_f=None):
        # Path of the image on which we are working.
        self.image_path = None
        # Image data corresponding to the `image_path`.
        self.input_image = None
        # Directory in which we export stuff relative to the `image_path`.
        self.working_directory = None
        # Path of the YOLO model that we use to classify microglia.
        self.classification_model_path = None
        # Path of the model that we use to segment microglia on YOLO patches.
        self.segmentation_model_path = None
        # Segmentation model.
        self.segmentation_model = None
        # Pixel size => tuple (pixel size, unit).
        self.calibration = None
        # Global logging function.
        self.logging = logging_f

    def log(self, message):
        if self.logging:
            self.logging(message)
    
    def create_working_directory(self, img_path):
        name = os.path.basename(img_path)
        wd_name = ".".join(name.split('.')[:-1])
        wd_name = wd_name.replace(" ", "-") + "-control"
        source_dir = os.path.dirname(img_path)
        self.working_directory = os.path.join(source_dir, wd_name)
        if os.path.isdir(self.working_directory):
            shutil.rmtree(self.working_directory)
        else:
            os.makedirs(self.working_directory, exist_ok=True)

    def load_image(self, image_path):
        self.image_path = image_path
        self.input_image = np.squeeze(tifffile.imread(image_path))
        self.create_working_directory(image_path)
        self.log(f"Image loaded: '{image_path}'")

    def get_image_path(self):
        return self.image_path
    
    def get_image_data(self):
        return self.input_image
    
    def get_working_directory(self):
        return self.working_directory
    
    def get_image_shape(self):
        return self.input_image.shape
    
    def set_calibration(self, pixel_size, unit):
        self.calibration = (pixel_size, unit)

    def set_segmentation_model_path(self, model_path):
        best_path = os.path.join(model_path, "best.keras")
        if not os.path.isfile(best_path):
            raise ValueError(f"Model '{os.path.basename(model_path)}' does not exist.")
        self.segmentation_model_path = best_path
        self.segmentation_model = tf.keras.models.load_model(self.segmentation_model_path)
        self.segmentation_model.summary()

    def set_classification_model_path(self, model_path):
        pass

    def export_patches(self):
        rescaled_img = recalibrate_image(self.input_image, *self.calibration)
        tiler = ImageTiler2D(_PATCH_SIZE, _OVERLAP, rescaled_img.shape)
        patches = tiler.image_to_tiles(rescaled_img)
        export_path = os.path.join(self.working_directory, "patches")
        shutil.rmtree(export_path, ignore_errors=True)
        os.makedirs(export_path, exist_ok=True)
        for i, patch in enumerate(patches):
            patch_name = f"patch_{str(i).zfill(3)}.tif"
            tifffile.imwrite(os.path.join(export_path, patch_name), patch)
        self.log(f"{len(patches)} patches exported to '{export_path}'")

    def segment_microglia(self):
        pass

    def classify_microglia(self):
        pass

    def make_skeletons(self):
        pass

    def extract_metrics(self):
        pass
