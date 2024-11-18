import os
import shutil
import pint
import json

import tifffile
import numpy as np
from skimage.measure import label, regionprops
from skimage import morphology

from microglia_analyzer.tiles.tiler import ImageTiler2D, normalize
from microglia_analyzer.dl.losses import dice_skeleton_loss, bce_dice_loss
from microglia_analyzer.utils import calculate_iou, normalize_batch, download_from_web

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import torch

class MicrogliaAnalyzer(object):
    
    def __init__(self, logging_f=None):
        # Image on which we are working.
        self.image = None
        # Pixel size => tuple (pixel size, unit).
        self.calibration = None
        # Directory in which we export productions (control images, settings, ...).
        self.working_directory = None
        # Path of the YOLO model that we use to classify microglia.
        self.classification_model_path = None
        # Path of the model that we use to segment microglia.
        self.segmentation_model_path = None
        # Classification model.
        self.classification_model = None
        # Segmentation model.
        self.segmentation_model = None
        # Importance of the skeleton in the loss function.
        self.unet_skeleton_coef = 0.2
        # Importance of the BCE in the BCE-dice loss function.
        self.unet_bce_coef = 0.7
        # Global logging function.
        self.logging = logging_f
        # Object responsible for cutting images into tiles.
        self.tiles_manager = None
        # Size of the tiles (in pixels).
        self.tile_size = None
        # Overlap between the tiles (in pixels).
        self.overlap = None
        # Probability threshold for the segmentation (%).
        self.segmentation_threshold = 0.5
        # Score threshold for the classification (%).
        self.score_threshold = 0.5
        # Probability map of the segmentation.
        self.probability_map = None
        # Connected component minimum size threshold.
        self.cc_min_size = 250
        # Classes guessed by the classification model.
        self.classes = None
        # Set of bounding-boxes guessed by the classification model.
        self.bboxes = None
        # Maximum IoU threshold (%) for the classification. Beyond that, BBs are merged.
        self.iou_threshold = 0.85
        # Bounding-boxes after they were cleaned.
        self.classifications = None
        # Dictionary of the bindings between the segmentation and the classification.
        self.bindings = None

    def log(self, message):
        if self.logging:
            self.logging(message)

    def set_input_image(self, image, tile_size=512, overlap=128):
        """
        Setter of the input image.
        Checks that the image is 2D before using it.
        """
        image = np.squeeze(image)
        if len(image.shape) != 2:
            raise ValueError("The input image must be 2D.")
        self.image = image
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiles_manager = ImageTiler2D(self.tile_size, self.overlap, image.shape)
    
    def set_calibration(self, pixel_size, unit):
        """
        Setter of the calibration.
        Before editing the internal state, checks that the pixel size is a float and Pint is used to check the unit.
        """
        if not isinstance(pixel_size, float):
            raise ValueError("The pixel size must be a float.")
        ureg = pint.UnitRegistry()
        try:
            unit = ureg(unit)
        except pint.UndefinedUnitError:
            raise ValueError("The unit is not recognized.")
        self.calibration = (pixel_size, unit)
    
    def set_working_directory(self, path):
        """
        Checks that the directory exists before setting it.
        Also outputs a warning in the logs if it is not empty.
        """
        if os.path.exists(path):
            self.log("The working directory already exists and will be cleared.")
            shutil.rmtree(path)
        os.makedirs(path)
        self.working_directory = path
    
    def set_segmentation_model(self, path, use="best"):
        """
        Checks that the path is a folder.
        In the folder, we search for the file "best.keras" or "last.keras".
        To verify that the training was complete, we also check for the presence of "training_history.png".
        """
        if not os.path.isdir(path):
            raise ValueError("The segmentation model path must be a folder.")
        model_path = os.path.join(path, use+".keras")
        if not os.path.exists(model_path):
            raise ValueError("The model file does not exist.")
        history_path = os.path.join(path, "training_history.png")
        if not os.path.exists(history_path):
            raise ValueError("The training of this model is not complete.")
        self.segmentation_model_path = path
        self.segmentation_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "bcl": bce_dice_loss(self.unet_bce_coef),
                "dsl": dice_skeleton_loss(self.unet_skeleton_coef, self.unet_bce_coef)
            }
        )

    def set_classification_model(self, path, use="best", reload=False):
        """
        Checks that the path corresponds to a folder.
        This folder must contain a "confusion_matrix.png" file to verify that the training is complete.
        In there, there must be a "weights" folder, containing either 'best.pt' or 'last.pt'.

        Args:
            - path (str): Path of the model's folder (containing 'results.csv' and 'weights').
            - use (str): Either 'best' or 'last', to use either 'best.pt' or 'last.pt'.
            - reload (bool): Whether to force the reload of the model from the online repo.
        """
        if not os.path.isdir(path):
            raise ValueError("The classification model path must be a folder.")
        weights_path = os.path.join(path, "weights", use+".pt")
        if not os.path.isfile(weights_path):
            raise ValueError("The model file does not exist.")
        confusion_matrix_path = os.path.join(path, "confusion_matrix.png")
        if not os.path.isfile(confusion_matrix_path):
            raise ValueError("The training of this model is not complete.")
        self.classification_model_path = weights_path
        self.classification_model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path=self.classification_model_path, 
            force_reload=reload
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classification_model.to(device)
    
    def segmentation_inference(self):
        tiles = np.array(self.tiles_manager.image_to_tiles(self.image))
        predictions = np.squeeze(self.segmentation_model.predict(tiles, batch_size=8))
        normalize_batch(predictions)
        self.probability_map = self.tiles_manager.tiles_to_image(predictions)

    def filter_cc_by_size(self, mask, connectivity=2):
        """
        Filters connected components in a binary mask based on their size.

        Args:
            - mask (np.ndarray): Binary mask (2D or 3D) with values 0 and 255.
            - min_size (int): Minimum number of pixels a connected component must have to be retained.
            - connectivity (int, optional): Connectivity criterion (4 or 8).

        Returns:
            (np.ndarray): Binary mask with only the connected components that meet the minimum size.
        """
        labeled_map = label(mask, connectivity=connectivity)
        regions = regionprops(labeled_map)
        labels_to_keep = [region.label for region in regions if region.area >= self.cc_min_size]

        if not labels_to_keep:
            return np.zeros_like(mask, dtype=np.uint8)

        filtered_binary = np.isin(labeled_map, labels_to_keep).astype(np.uint8) * 255
        return filtered_binary

    def segmentation_postprocess(self):
        self.mask = (self.probability_map > self.segmentation_threshold).astype(np.uint8)
        self.mask = self.filter_cc_by_size(self.mask)
        selem = morphology.diamond(2)
        self.mask = morphology.binary_closing(self.mask, selem)
        # self.mask = morphology.binary_fill_holes(self.mask)
        self.mask = label(self.mask, connectivity=2)
        
    def classification_inference(self):
        yolo_input = normalize(self.image, 0, 255, np.uint8)
        results = self.classification_model(yolo_input)
        for img_results in results.xyxy:
            boxes   = img_results[:, :4].tolist()
            scores  = img_results[:, 4].tolist()
            classes = img_results[:, 5].tolist()
            self.bboxes = {
                'boxes'  : boxes,
                'scores' : scores,
                'classes': classes,
            }
    
    def classification_postprocess(self):
        """
        Fusions boxes with an IoU greater than `iou_threshold`.
        The box with the highest score is kept, whatever the two classes were.
        Also, boxes with a score below the threshold score are removed.

        Parameters:
        - boxes: list of dict, chaque dict contient 'box' (coordonnées) et 'class'
        - iou_threshold: float, seuil d'IoU pour fusionner les boîtes

        Returns:
        - fused_boxes: list of dict, boîtes après fusion
        """
        clean_boxes = {'boxes': [], 'scores': [], 'classes': []}
        used        = set()

        for i, (box1, score1, class1) in enumerate(zip(self.bboxes['boxes'], self.bboxes['scores'], self.bboxes['classes'])):
            if i in used:
                continue
            chosen_box = box1
            chosen_score = score1
            chosen_class = class1
            for j, (box2, score2, class2) in enumerate(zip(self.bboxes['boxes'], self.bboxes['scores'], self.bboxes['classes'])):
                if j <= i or j in used:
                    continue
                iou = calculate_iou(chosen_box, box2)
                if iou > self.iou_threshold:
                    chosen_box   = chosen_box if score1 > score2 else box2
                    chosen_score = max(score1, score2)
                    chosen_class = class1 if score1 > score2 else class2
                    used.add(j)
            if chosen_score < self.score_threshold:
                continue
            clean_boxes['boxes'].append(chosen_box)
            clean_boxes['scores'].append(chosen_score)
            clean_boxes['classes'].append(chosen_class)
            used.add(i)
        self.classifications = clean_boxes
    
    def bind_classifications(self):
        labeled = self.mask
        regions = regionprops(labeled)
        bindings = {int(l): (None, 0.0, None) for l in np.unique(labeled) if l != 0} # label: (class, IoU)
        for region in regions:
            seg_bbox = list(map(int, region.bbox))
            for box, cls in zip(self.classifications['boxes'], self.classifications['classes']):
                detect_bbox = list(map(int, box))
                iou = calculate_iou(seg_bbox, detect_bbox)
                if iou > bindings[region.label][1]:
                    bindings[region.label] = (cls, iou, seg_bbox)
        self.bindings = bindings


if __name__ == "__main__":
    img_path = "/home/benedetti/Documents/projects/2060-microglia/data/raw-data/tiff-data/adulte 3.tif"
    img_data = tifffile.imread(img_path)
    ma = MicrogliaAnalyzer(lambda x: print(x))
    ma.set_input_image(img_data)
    ma.set_calibration(0.325, "µm")
    ma.set_segmentation_model("/home/benedetti/Documents/projects/2060-microglia/µnet/µnet-V208")
    ma.set_classification_model("/home/benedetti/Documents/projects/2060-microglia/µyolo/µyolo-V051")
    ma.segmentation_inference()
    ma.segmentation_postprocess()
    ma.classification_inference()
    ma.classification_postprocess()
    ma.bind_classifications()