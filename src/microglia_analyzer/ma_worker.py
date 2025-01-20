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

# Size of the tiles (in pixels).
_UNET_TILE    = 512
# _UNET_OVERLAP between the tiles (in pixels).
_UNET_OVERLAP = 128

# Size of the tiles (in pixels).
_YOLO_TILE    = 512
# Overlap for YOLO
_YOLO_OVERLAP = 256


class MicrogliaAnalyzer(object):
    
    def __init__(self, logging_f=None):
        # Global logging function.
        self.logging = logging_f
        # Image on which we are working.
        self.image = None
        # Pixel size => tuple (pixel size, unit).
        self.calibration = None
        # Directory in which we export productions (control images, settings, ...).
        self.working_directory = None
        # Path of the model that we use to segment microglia.
        self.segmentation_model_path = None
        # Segmentation model.
        self.segmentation_model = None
        # Probability map of the segmentation.
        self.probability_map = None
        # Probability threshold for the segmentation (%).
        self.segmentation_threshold = 0.5
        # Minimal area (in µm²) to cover for an element to be considered.
        self.min_surface = 250
        # Final mask of the segmentation.
        self.mask = None
        # Names of classes guessed by the classification model.
        self.class_names = None
        # Path of the YOLO model that we use to classify microglia.
        self.classification_model_path = None
        # Classification model.
        self.classification_model = None
        # Set of bounding-boxes, classes and scores guessed by the classification model.
        self.yolo_output = None
        # Dictionary of the bindings between the segmentation and the classification.
        self.bindings = None
        # Graph metrics extracted from each label
        self.graph_metrics = None
        # Skeleton of the segmentation.
        self.skeleton = None

    def _log(self, message):
        if self.logging:
            self.logging(message)

    def reset_segmentation(self):
        self.image = None
        self.probability_map = None
        self.mask = None

    def reset_classification(self):
        self.yolo_output = None
        self.bindings = None
        self.graph_metrics = None
        self.skeleton = None

    def set_input_image(self, image):
        """
        Setter of the input image.
        Checks that the image is 2D before using it.
        """
        self.reset_segmentation()
        self.reset_classification()
        image = np.squeeze(image)
        if len(image.shape) != 2:
            raise ValueError("The input image must be 2D.")
        self.image = image
        self._log(f"New image shape: {self.image.shape}")
    
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
        self._log(f"Calibration set to: {self.calibration[0]} {self.calibration[1]}")
    
    def set_working_directory(self, path):
        """
        Checks that the directory exists before setting it.
        Also outputs a warning in the logs if it is not empty.
        """
        if os.path.exists(path):
            self._log("The working directory already exists and will be cleared.")
            shutil.rmtree(path)
        os.makedirs(path)
        self.working_directory = path
        self._log(f"Working directory set to: {self.working_directory}")
    
    def set_segmentation_model(self, path, use="best"):
        """
        Checks that the path is a folder.
        In the folder, we search for the file "best.keras" or "last.keras".
        To verify that the training was complete, we also check for the presence of "training_history.png".
        """
        if not os.path.isdir(path):
            raise ValueError("The segmentation model path must be a folder.")
        model_path = os.path.join(path, use+".keras")
        self._log(f"Searching for model: {model_path}...")
        if not os.path.isfile(model_path):
            raise ValueError("The model file does not exist.")
        history_path = os.path.join(path, "training_history.png")
        if not os.path.exists(history_path):
            raise ValueError("The training of this model is not complete.")
        self.segmentation_model_path = model_path
        self.segmentation_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "bcl": bce_dice_loss(0.0),
                "dsl": dice_skeleton_loss(0.0, 0.0),
                "dice_skeleton_loss": dsl
            }
        )
        self._log("Segmentation model path set to: " + str(self.segmentation_model_path))

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
        self._log(f"Searching for model: {weights_path}...")
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
        dn = self.classification_model.names
        self.class_names = [dn[int(i)] for i in range(len(dn))]
        self._log("Classification model path set to: " + str(self.classification_model_path))
        self._log("Classes found: " + str(self.class_names))

    def _segmentation_inference(self):
        shape = self.image.shape
        tiles_manager = ImageTiler2D(_UNET_TILE, _UNET_OVERLAP, shape)
        input_unet = normalize(self.image, 0.0, 1.0, np.float32)
        tiles = np.array(tiles_manager.image_to_tiles(input_unet, False))
        predictions = np.squeeze(self.segmentation_model.predict(tiles, batch_size=8))
        # normalize_batch(predictions)
        self.probability_map = tiles_manager.tiles_to_image(predictions)

    def set_proba_threshold(self, threshold):
        if not 0.0 <= threshold <= 1.0:
            print("The probability threshold must be between 0 and 1.")
            return
        self.segmentation_threshold = threshold
        self._log(f"Probability threshold set to: {self.segmentation_threshold}")
        self._segmentation_postprocessing()

    def set_min_surface(self, min_size):
        if min_size < 0:
            print("The minimum size must be a positive integer.")
            return
        self.min_surface = min_size
        self._segmentation_postprocessing()
        self._log(f"Minimum surface set to: {self.min_surface} µm²")

    def _filter_by_area(self, mask, connectivity=2):
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
        n_pixels = int(self.min_surface / (self.calibration[0]**2))
        print(f"Removed items smaller than {self.min_surface} µm² ({n_pixels} pixels)")
        labels_to_keep = [region.label for region in regions if region.area >= n_pixels]
        if not labels_to_keep:
            return np.zeros_like(mask, dtype=np.uint8)
        filtered_binary = np.isin(labeled_map, labels_to_keep).astype(np.uint8)
        return filtered_binary

    def _segmentation_postprocessing(self):
        if self.probability_map is None:
            return
        # Filter probability map by threshold.
        self.mask = self.probability_map > self.segmentation_threshold
        # Trying to merge islands.
        selem = morphology.disk(3)
        self.mask = morphology.binary_closing(self.mask, selem)
        # Filter by area.
        self.mask = self._filter_by_area(self.mask)
        # Connected components labeling.
        self.mask = label(self.mask, connectivity=2)

    def segment_microglia(self):
        self._segmentation_inference()
        self._segmentation_postprocessing()

    def get_mask(self, show_garbage):
        if self.mask is None:
            return None
        if show_garbage:
            return (self.mask > 0).astype(np.uint8) * 255
        if self.bindings is None:
            return (self.mask > 0).astype(np.uint8) * 255
        garbages = [i for i, (c, b) in enumerate(self.bindings) if (c == 0) and (i != 0)]
        to_be_removed = np.isin(self.mask, garbages)
        clean_mask = np.copy(self.mask)
        clean_mask[to_be_removed] = 0
        return (clean_mask > 0).astype(np.uint8) * 255

    def _classification_inference(self):
        self.reset_classification()
        yolo_input = normalize(self.image, 0, 255, np.uint8)
        tiles_manager = ImageTiler2D(_YOLO_TILE, _YOLO_OVERLAP, self.image.shape)
        tiles = tiles_manager.image_to_tiles(yolo_input, False)
        results = self.classification_model(tiles)
        self.yolo_output = []
        for i, img_results in enumerate(results.xyxy):
            boxes   = img_results[:, :4].tolist()
            y, x    = tiles_manager.layout[i].ul_corner
            boxes   = [[box[1] + y, box[0] + x, box[3] + y, box[2] + x] for box in boxes]
            scores  = img_results[:, 4].tolist()
            classes = [int(c) for c in img_results[:, 5].tolist()]
            for box, score, c in zip(boxes, scores, classes):
                self.yolo_output.append((box, score, c))

    def _bind_classifications(self, votes):
        self.bindings = [(0, (0, 0, 0, 0)) for _ in range(len(votes))]
        all_props     = regionprops(self.mask)
        for props in all_props:
            lbl = props.label
            if lbl == 0:
                continue
            y1, x1, y2, x2 = props.bbox
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
            max_vote = np.argmax(votes[lbl])
            self.bindings[lbl] = (max_vote, (y1, x1, y2, x2))

    def _classification_postprocessing(self, skip_garbage=False):
        if (self.mask is None) or (self.yolo_output is None):
            return
        labeled = self.mask
        votes = np.zeros((np.max(labeled)+1, len(self.class_names)), dtype=np.float32)
        for box, score, cls in self.yolo_output:
            if skip_garbage and (cls == 0):
                continue
            y1, x1, y2, x2 = box
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
            sub_image = labeled[y1:y2, x1:x2]
            counts = np.bincount(sub_image.flatten())
            counts[0] = 0 # We don't want to count the background
            target_lbl = int(np.argmax(counts))
            coef = counts[target_lbl] * score
            votes[target_lbl, cls] += coef
        self._bind_classifications(votes)

    def classify_microglia(self):
        self._classification_inference()
        self._classification_postprocessing()

    def _sort_labels_by_class(self):
        expended_bindings = [(i, cls) for (i, (cls, _)) in enumerate(self.bindings) if (cls != 0) and (i != 0)]
        sorted_labels = sorted(expended_bindings, key=lambda x: x[1])
        return sorted_labels

    def _analyze_skeleton(self, mask):
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
            "# branches": num_branches,
            "# leaves": num_leaves,
            "# junctions": num_junctions,
            "Average branch length": round(avg_branch_length, 2),
            "Total length": round(total_length, 2),
            "Max branch length": round(max_branch_length, 2)
        }

        return results, skeleton

    def analyze_graph(self):
        labels     = self._sort_labels_by_class()
        results    = []
        skeletons  = np.zeros_like(self.mask, dtype=np.uint8)
        for label, cls in labels:
            mask = (self.mask == label).astype(np.uint8)
            try:
                r, s = self._analyze_skeleton(mask)
                r['Label'] = label
                r['Class'] = self.class_names[cls]
            except:
                continue
            results.append(r)
            skeletons = np.maximum(skeletons, (s > 0).astype(np.uint8))
        self.graph_metrics = results
        self.skeleton = skeletons

    def as_tsv(self, identifier):
        if len(self.graph_metrics) == 0:
            return None
        first_label = self.graph_metrics[0]
        headers = ["Source"] + list(first_label.keys())
        buffer  = ["\t ".join(headers)]

        for i, measures in enumerate(self.graph_metrics):
            values = ["" if i > 0 else identifier] + [str(measures[key]) for key in measures.keys()]
            line = "\t ".join([str(v) for v in values])
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
    ma.segment_microglia()
    ma.classify_microglia()
    ma.analyze_graph()
    tsv = ma.as_tsv("adulte 3")
    with open("/tmp/metrics.csv", "w") as f:
        f.write("\n".join(tsv))
