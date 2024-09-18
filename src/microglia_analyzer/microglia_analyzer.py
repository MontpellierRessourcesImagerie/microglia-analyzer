import tifffile
import numpy as np
import os

class MicrogliaAnalyzer(object):
    
    def __init__(self, logging_f=None):
        # Path of the image on which we are working.
        self.image_path = None
        # Image data corresponding to the `image_path`.
        self.input_image = None
        # Directory in which we export stuff relative to the `image_path`.
        self.working_directory = None
        # Name of the YOLO model that we use to detect microglia.
        self.detection_model = None
        # Name of the model that we use to segment microglia on YOLO patches.
        self.segmentation_model = None
        # Pixel size (in µm).
        self.calibration = None
        # Global logging function.
        self.logging = logging_f

    def log(self, message):
        if self.logging:
            self.logging(message)

    def get_image_path(self):
        return self.image_path
    
    def get_image_data(self):
        return self.input_image
    
    def get_working_directory(self):
        return self.working_directory
    
    def get_image_shape(self):
        return self.input_image.shape

    def load_image(self, image_path):
        self.image_path = image_path
        self.input_image = np.squeeze(tifffile.imread(image_path))
        name = os.path.basename(image_path)
        wd_name = ".".join(name.split('.')[:-1])
        wd_name = wd_name.replace(" ", "-")
        wd_name += "-control"
        source_dir = os.path.dirname(image_path)
        self.working_directory = os.path.join(source_dir, wd_name)
        os.makedirs(self.working_directory, exist_ok=True)
        self.log(f"Image loaded: '{name}'")
