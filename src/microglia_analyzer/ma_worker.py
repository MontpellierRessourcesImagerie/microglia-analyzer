import os
import shutil
import pint
import json
import pathlib
import platform

import tifffile
import numpy as np
from skimage.measure import label, regionprops
from skimage import morphology
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

from microglia_analyzer.tiles.tiler import normalize
from microglia_analyzer.utils import calculate_iou, normalize_batch
from microglia_analyzer.tiles.tiler import ImageTiler2D

os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import torch
import tensorflow as tf
from microglia_analyzer.dl.losses import dice_skeleton_loss, bce_dice_loss, dsl

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
        # Segmentation model.
        self.segmentation_model = None
        # Classification model.
        self.classification_model = None
        # Global logging function.
        self.logging = logging_f
        # Object responsible for cutting images into tiles.
        self.tiles_manager = None
        # Size of the tiles (in pixels).
        self.unet_tile_size = None
        # Size of the tiles (in pixels).
        self.yolo_tile_size = None
        # Overlap for YOLO
        self.yolo_overlap = None
        # unet_overlap between the tiles (in pixels).
        self.unet_overlap = None
        # Probability threshold for the segmentation (%).
        self.segmentation_threshold = 0.5
        # Importance of the skeleton in the loss function.
        self.unet_skeleton_coef = 0.2
        # Importance of the BCE in the BCE-dice loss function.
        self.unet_bce_coef = 0.7
        # Score threshold for the classification (%).
        self.score_threshold = 0.35
        # Probability map of the segmentation.
        self.probability_map = None
        # Connected component minimum size threshold.
        self.cc_min_size = 250
        # Classes guessed by the classification model.
        self.classes = None
        # Set of bounding-boxes guessed by the classification model.
        self.bboxes = None
        # Maximum IoU threshold (%) for the classification. Beyond that, BBs are merged.
        self.iou_threshold = 0.25
        # Bounding-boxes after they were cleaned.
        self.classifications = None
        # Dictionary of the bindings between the segmentation and the classification.
        self.bindings = None
        # Final mask of the segmentation.
        self.mask = None
        # Graph metrics extracted from each label
        self.graph_metrics = None
        # Skeleton of the segmentation.
        self.skeleton = None

    def log(self, message):
        if self.logging:
            self.logging(message)

    def set_input_image(self, image, unet_tile_size=512, unet_overlap=128, yolo_tile_size=640, yolo_overlap=128):
        """
        Setter of the input image.
        Checks that the image is 2D before using it.
        """
        image = np.squeeze(image)
        if len(image.shape) != 2:
            raise ValueError("The input image must be 2D.")
        self.image = image
        self.unet_tile_size = unet_tile_size
        self.unet_overlap = unet_overlap
        self.yolo_tile_size = yolo_tile_size
        self.yolo_overlap = yolo_overlap
    
    def set_calibration(self, pixel_size, unit):
        """
        Setter of the calibration.
        Before editing the internal state, checks that the pixel size is a float and Pint is used to check the unit.
        """
        if not isinstance(pixel_size, float):
            raise ValueError("The pixel size must be a float.")
        ureg = pint.UnitRegistry()
        try:
            ureg(unit)
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
        self.segmentation_model_path = model_path
        print("Segmentation model path set to: ", self.segmentation_model_path)
        self.segmentation_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "bcl": bce_dice_loss(self.unet_bce_coef),
                "dsl": dice_skeleton_loss(self.unet_skeleton_coef, self.unet_bce_coef),
                "dice_skeleton_loss": dsl
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
        plt = platform.system()
        if plt == "Windows":
            pathlib.PosixPath = pathlib.WindowsPath
        self.classification_model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path=self.classification_model_path, 
            force_reload=reload
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classification_model.to(device)
        self.classes = self.classification_model.names

    def segmentation_inference(self):
        shape = self.image.shape
        tiles_manager = ImageTiler2D(self.unet_tile_size, self.unet_overlap, shape)
        input_unet = normalize(self.image, 0, 1, np.float32)
        tiles = np.array(tiles_manager.image_to_tiles(input_unet, False))
        predictions = np.squeeze(self.segmentation_model.predict(tiles, batch_size=8))
        normalize_batch(predictions)
        self.probability_map = tiles_manager.tiles_to_image(predictions)

    def set_proba_threshold(self, threshold):
        if not 0.0 <= threshold <= 1.0:
            print("The probability threshold must be between 0 and 1.")
            return
        self.segmentation_threshold = threshold

    def set_cc_min_size(self, min_size):
        if min_size < 0:
            print("The minimum size must be a positive integer.")
            return
        self.cc_min_size = min_size

    def set_min_score(self, min_score):
        if not 0.0 <= min_score <= 1.0:
            print("The minimum score must be between 0 and 1.")
            return
        self.score_threshold = min_score

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

    def segmentation_postprocessing(self):
        self.mask = (self.probability_map > self.segmentation_threshold).astype(np.uint8)
        self.mask = self.filter_cc_by_size(self.mask)
        selem = morphology.diamond(4)
        self.mask = morphology.binary_closing(self.mask, selem)
        # self.mask = morphology.binary_fill_holes(self.mask)
        self.mask = label(self.mask, connectivity=2)
    
    def _filter_garbage(self, garbage=0):
        filtered_bboxes = {
            'boxes': [],
            'scores': [],
            'classes': []
        }
        for box, score, cls in zip(self.bboxes['boxes'], self.bboxes['scores'], self.bboxes['classes']):
            if cls == garbage:
                continue
            filtered_bboxes['boxes'].append(box)
            filtered_bboxes['scores'].append(score)
            filtered_bboxes['classes'].append(cls)
        self.bboxes = filtered_bboxes

    def classification_inference(self, remove_garbage=True):
        yolo_input = normalize(self.image.copy(), 0, 255, np.uint8)
        tiles_manager = ImageTiler2D(self.yolo_tile_size, self.yolo_overlap, self.image.shape)
        tiles = tiles_manager.image_to_tiles(yolo_input, False)
        results = self.classification_model(tiles)
        self.bboxes = {'boxes': [], 'scores': [], 'classes': []}
        for i, img_results in enumerate(results.xyxy):
            boxes   = img_results[:, :4].tolist()
            y, x = tiles_manager.layout[i].ul_corner
            boxes   = [[box[0] + x, box[1] + y, box[2] + x, box[3] + y] for box in boxes]
            scores  = img_results[:, 4].tolist()
            classes = img_results[:, 5].tolist()
            self.bboxes['boxes'] += boxes
            self.bboxes['scores'] += scores
            self.bboxes['classes'] += classes
        if remove_garbage:
            self._filter_garbage(0)
    
    def classification_postprocessing(self):
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
    
    def _bind_classifications(self):
        labeled = self.mask
        regions = regionprops(labeled)
        bindings = {int(l): (None, 0.0, None) for l in np.unique(labeled) if l != 0} # label: (class, IoU)
        for region in regions:
            seg_bbox = list(map(int, region.bbox))
            bindings[region.label] = (0, 0.0, seg_bbox)
            for box, cls in zip(self.classifications['boxes'], self.classifications['classes']):
                x1, y1, x2, y2 = list(map(int, box))
                detect_bbox = [y1, x1, y2, x2]
                iou = calculate_iou(seg_bbox, detect_bbox)
                if iou > bindings[region.label][1]: # iou > 0.2 and
                    bindings[region.label] = (cls, iou, seg_bbox)
        self.bindings = bindings
    
    def bind_classifications(self):
        labeled = self.mask
        regions = regionprops(labeled)
        # box_index: (class, IoU, label, seg_bbox)
        bindings = {b: (None, 0.0, None, None) for b in range(len(self.classifications['boxes']))}
        
        for b_index, (box, cls) in enumerate(zip(self.classifications['boxes'], self.classifications['classes'])):
            x1, y1, x2, y2 = list(map(int, box))
            detect_bbox = [y1, x1, y2, x2]
            for region in regions:
                seg_bbox = list(map(int, region.bbox))
                iou = calculate_iou(seg_bbox, detect_bbox)
                if iou > bindings[b_index][1]:
                    bindings[b_index] = (cls, iou, region.label, seg_bbox)
                
        self.bindings = self.flip_dict(bindings)
    
    def flip_dict(self, d):
        flipped = {}
        for _, (cls, iou, label, seg_bbox) in d.items():
            if seg_bbox is None:
                continue
            flipped[label] = (cls, iou, seg_bbox)
        return flipped

    def analyze_skeleton(self, mask):
        skeleton = skeletonize(mask)
        skel = Skeleton(skeleton)
        branch_data = summarize(skel, separator='_')
        factor = self.calibration[0] if self.calibration else 1.0

        num_branches      = len(branch_data)
        num_leaves        = np.sum(branch_data['branch_type'] == 1)
        num_junctions     = np.sum(branch_data['branch_type'] == 2)
        avg_branch_length = np.mean(branch_data['branch_distance']) * factor
        total_length      = branch_data['branch_distance'].sum()    * factor
        max_branch_length = branch_data['branch_distance'].max()    * factor

        results = {
            "number_of_branches"   : num_branches,
            "number_of_leaves"     : num_leaves,
            "number_of_junctions"  : num_junctions,
            "average_branch_length": round(avg_branch_length, 2),
            "total_length"         : round(total_length, 2),
            "max_branch_length"    : round(max_branch_length, 2)
        }

        return results, skeleton

    def analyze_as_graph(self):
        labels    = np.unique(self.mask)
        results   = {}
        skeletons = np.zeros_like(self.mask)
        for label in labels:
            if label == 0:
                continue
            mask = (self.mask == label).astype(np.uint8)
            results[label], skeleton = self.analyze_skeleton(mask)
            skeletons = np.maximum(skeletons, skeleton)
        self.graph_metrics = results
        self.skeleton = skeletons
    
    def sorted_by_class(self, bindings, common_labels):
        sorted_bindings = {}
        for label, (cls, iou, seg_bbox) in bindings.items():
            if label not in common_labels:
                continue
            if cls not in sorted_bindings:
                sorted_bindings[cls] = []
            sorted_bindings[cls].append((label, iou, seg_bbox))
        return sorted_bindings

    def sort_labels_by_class(self, data, valid_labels):
        filtered = {label: value for label, value in data.items() if label in valid_labels}
        sorted_labels = sorted(filtered.keys(), key=lambda label: filtered[label][0])
        return sorted_labels

    def as_csv(self, identifier):
        common_labels = set(self.graph_metrics.keys()) & set(self.bindings.keys())
        if len(common_labels) == 0:
            return None
        sorted_labels = self.sort_labels_by_class(self.bindings, common_labels)
        
        first_label = sorted_labels[0]
        graph_measure_keys = list(self.graph_metrics[first_label].keys())
        headers = ["Identifier"] + graph_measure_keys + ["IoU", "Class"]
        buffer = [", ".join(headers)]

        for i, label in enumerate(sorted_labels):
            values = [""]
            if i == 0:
                values[0] = identifier
            graph_measures = self.graph_metrics[label]
            class_value, iou = self.bindings[label][:2]
            class_value = self.classes[int(class_value)] if class_value is not None else ""
            values += [graph_measures[key] for key in graph_measure_keys] + [iou, class_value]
            line = ", ".join([str(v) for v in values])
            buffer.append(line)

        return buffer


if __name__ == "__main__":
    img_path = "/home/benedetti/Documents/projects/2060-microglia/data/raw-data/tiff-data/adulte 3.tif"
    img_data = tifffile.imread(img_path)
    ma = MicrogliaAnalyzer(lambda x: print(x))
    ma.set_input_image(img_data)
    ma.set_calibration(0.325, "µm")
    ma.set_segmentation_model("/home/benedetti/Documents/projects/2060-microglia/µnet/µnet-V208")
    ma.set_classification_model("/home/benedetti/Documents/projects/2060-microglia/µyolo/µyolo-V051")
    ma.segmentation_inference()
    ma.segmentation_postprocessing()
    ma.classification_inference()
    ma.classification_postprocessing()
    ma.bind_classifications()
    ma.analyze_as_graph()
    csv = ma.as_csv("adulte 3")
    with open("/tmp/metrics.csv", "w") as f:
        f.write("\n".join(csv))
