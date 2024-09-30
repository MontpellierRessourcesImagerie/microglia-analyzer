from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLineEdit,
                            QHBoxLayout, QPushButton, QLabel,
                            QFileDialog, QComboBox, QGroupBox)

from napari.utils.notifications import show_info

import tifffile
from microglia_analyzer import TIFF_REGEX

import numpy as np
import os

# Prefix of a layer name to be considered as a YOLO class.
_CLASS_PREFIX = "class."
# Name of the layer containing the current image.
_IMAGE_LAYER  = "Image"
# Colors assigned to each YOLO class.
_COLORS       = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#FF8000",
    "#8000FF",
    "#0080FF",
    "#80FF00",
    "#FF0080",
    "#00FF80",
    "#800000",
    "#008000",
    "#800080",
    "#808000"
]

# A YOLO bounding-box == a tuple of 5 elements:
#   - (int) The class to which this box belongs.
#   - (float) x component of the box's center (between 0.0 and 1.0) in percentage of image width.
#   - (float) y component of the box's center (between 0.0 and 1.0) in percentage of image height.
#   - (float) width of the box (between 0.0 and 1.0) in percentage of image width.
#   - (float) height of the box (between 0.0 and 1.0) in percentage of image height.

class AnnotateBoundingBoxesWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        # Active Napari viewer.
        self.viewer = napari_viewer
        # Folder in which are located the 'images' and 'annotations' folders.
        self.root_directory = None
        # Name of the folder in which images are located
        self.sources_directory = None
        # Name of the folder in which annotations are located
        self.annotations_directory = None
        # List of images ('.tif') in the 'images' folder.
        self.images_list = []

        self.layout = QVBoxLayout()
        self.init_ui()
        self.setLayout(self.layout)

    # -------- UI: ----------------------------------

    def add_media_management_group_ui(self):
        box = QGroupBox("Media management")
        layout = QVBoxLayout()
        box.setLayout(layout)

        # Label + button to select the source directory:
        self.select_sources_directory_button = QPushButton("📂 Sources directory")
        self.select_sources_directory_button.clicked.connect(self.select_sources_directory)
        layout.addWidget(self.select_sources_directory_button)

        # Label + text box for the inputs sub-folder's name:
        inputs_name_label = QLabel("Inputs sub-folder:")
        self.inputs_name = QComboBox()
        self.inputs_name.currentIndexChanged.connect(self.set_sources_directory)
        self.inputs_name.addItem("---")
        h_layout = QHBoxLayout()
        h_layout.addWidget(inputs_name_label)
        h_layout.addWidget(self.inputs_name)
        layout.addLayout(h_layout)

        # Label + text box for the annotations sub-folder's name:
        annotations_name_label = QLabel("Annotations sub-folder:")
        self.annotations_name = QLabel()
        self.annotations_name.setText("---")
        self.annotations_name.setMaximumHeight(20)
        font = self.annotations_name.font()
        font.setBold(True)
        self.annotations_name.setFont(font)
        h_layout = QHBoxLayout()
        h_layout.addWidget(annotations_name_label)
        h_layout.addWidget(self.annotations_name)
        layout.addLayout(h_layout)

        self.layout.addWidget(box)
    
    def add_classes_management_group_ui(self):
        box = QGroupBox("Classes management")
        layout = QVBoxLayout()
        box.setLayout(layout)

        # Adds a new class with the current name in the text box:
        h_laytout = QHBoxLayout()
        self.new_name = QLineEdit()
        h_laytout.addWidget(self.new_name)
        self.add_yolo_class_button = QPushButton("🔖 New class")
        self.add_yolo_class_button.clicked.connect(self.add_yolo_class)
        h_laytout.addWidget(self.add_yolo_class_button)
        layout.addLayout(h_laytout)

        # Label showing the number of boxes in each class
        self.counts_label = QLabel("Counts:")
        font = self.counts_label.font()
        font.setBold(True)
        self.counts_label.setFont(font)
        layout.addWidget(self.counts_label)
        self.count_display_label = QLabel("")
        self.counts_label.setMaximumHeight(20)
        self.count_display_label.setMaximumHeight(10)
        layout.addWidget(self.count_display_label)

        self.layout.addWidget(box)
    
    def add_annotations_management_group_ui(self):
        box = QGroupBox("Annotations management")
        layout = QVBoxLayout()
        box.setLayout(layout)

        # Label + combobox containing inputs list:
        self.image_selector = QComboBox()
        self.image_selector.currentIndexChanged.connect(self.open_image)
        self.image_selector.addItem("---")
        layout.addWidget(self.image_selector)

        # Button to save the annotations:
        self.save_button = QPushButton("💾 Save annotations")
        self.save_button.clicked.connect(self.save_state)
        layout.addWidget(self.save_button)

        self.layout.addWidget(box)

    def init_ui(self):
        self.add_media_management_group_ui()
        self.add_classes_management_group_ui()
        self.add_annotations_management_group_ui()

    # ----------------- CALLBACKS -------------------------------------------

    def select_sources_directory(self):
        """
        Select the folder containing the "images" and "labels" sub-folders.
        """
        directory = QFileDialog.getExistingDirectory(self, "Select root directory")
        if not os.path.isdir(directory):
            show_info("Invalid directory.")
            return
        self.set_root_directory(directory)

    def upper_corner(self, box):    
        """
        Locates the upper-right corner of a bounding-box having the Napari format.
        Works only for rectangles.
        The order of coordinates changes depending on the drawing direction of the rectangle!!!

        Args:
            - box (str): A Napari bounding-box, as it can be found in the '.data' of a shape layer.
        """
        return [np.max(axis) for axis in box.T]
    
    def lower_corner(self, box):
        """
        Locates the lower-left corner of a bounding-box having the Napari format.
        Works only for rectangles.
        The order of coordinates changes depending on the drawing direction of the rectangle!!!

        Args:
            - box (str): A Napari bounding-box, as it can be found in the '.data' of a shape layer.
        """
        return [np.min(axis) for axis in box.T]

    def yolo2bbox(self, bboxes):
        """
        Takes a YOLO bounding-box and converts it to the Napari format.
        The output can be used in a shape layer.
        """
        xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
        xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
        return xmin, ymin, xmax, ymax
    
    def bbox2yolo(self, bbox):
        """
        Converts a Napari bounding-box (from a shape layer) into a YOLO bounding-box.
        Doesn't handle the class (int) by itself, only the coordinates.

        Args:
            - bbox (np.array): Array containing coordinates of a bounding-box as in a shape layer.
        
        Returns:
            (tuple): 4 floats representing the x-centroid, the y-centroid, the width and the height in YOLO format.
                     It means that all these coordinates are percentages of the image's dimensions.
        """
        ymax, xmax = self.upper_corner(bbox)
        ymin, xmin = self.lower_corner(bbox)
        height, width = self.viewer.layers[_IMAGE_LAYER].data.shape[:2]
        x = (xmin + xmax) / 2 / width
        y = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        return round(x, 3), round(y, 3), round(w, 3), round(h, 3)
    
    def layer2yolo(self, layer_name, index):
        data = self.viewer.layers[layer_name].data
        tuples = []
        for rectangle in data:
            bbox = (index,) + self.bbox2yolo(rectangle)
            tuples.append(bbox)
        return tuples
    
    def write_annotations(self, tuples):
        labels_folder = os.path.join(self.root_directory, self.annotations_directory)
        current_as_txt = TIFF_REGEX.match(self.image_selector.currentText()).group(1) + ".txt"
        labels_path = os.path.join(labels_folder, current_as_txt)
        with open(labels_path, "w") as f:
            for row in tuples:
                f.write(" ".join(map(str, row)) + "\n")
        with open(os.path.join(self.root_directory, "classes.txt"), "w") as f:
            for c in self.get_classes():
                f.write(c + "\n")
        show_info("Annotations saved.")
    
    def save_state(self):
        count = 0
        lines = []
        for l in self.viewer.layers:
            if not l.name.startswith(_CLASS_PREFIX):
                continue
            lines += self.layer2yolo(l.name, count)
            count += 1
        self.write_annotations(lines)
        self.count_boxes()

    def get_new_class_name(self):
        """
        Probes the input text box and returns the name of the new class.
        """
        name_candidate = self.new_name.text().lower().replace(" ", "-")
        if name_candidate == "":
            show_info("Empty name.")
            return None
        if name_candidate.startswith(_CLASS_PREFIX):
            full_name = name_candidate
        else:
            full_name = _CLASS_PREFIX + name_candidate
        self.new_name.setText("")
        return full_name

    def add_yolo_class(self):
        class_name = self.get_new_class_name()
        if _IMAGE_LAYER not in self.viewer.layers:
            show_info("No image loaded.")
            return
        n_classes = len(self.get_classes())
        color = _COLORS[n_classes % len(_COLORS)]
        l = self.viewer.layers.selection.active
        if class_name is None:
            return
        # Clears the selection of the current layer.
        if (l is not None) and (l.name.startswith(_CLASS_PREFIX)):
            l.selected_data = set()
        self.viewer.add_shapes(
            name=class_name,
            edge_color=color,
            face_color="transparent",
            opacity=0.8,
            edge_width=3
        )
    
    def set_root_directory(self, directory):
        folders = sorted([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])
        folders = ["---"] + folders
        self.inputs_name.clear()
        self.inputs_name.addItems(folders)
        self.root_directory = directory

    def set_sources_directory(self):
        source_folder = self.inputs_name.currentText()
        annotations_folder = source_folder + "-labels"
        if (source_folder is None) or (source_folder == "---") or (source_folder == ""):
            return
        inputs_path      = os.path.join(self.root_directory, source_folder)
        annotations_path = os.path.join(self.root_directory, annotations_folder)
        if not os.path.isdir(inputs_path):
            return False
        if not os.path.isdir(annotations_path):
            os.makedirs(annotations_path)
        self.sources_directory     = source_folder
        self.annotations_directory = annotations_folder
        self.annotations_name.setText(annotations_folder)
        self.open_sources_directory()
    
    def open_sources_directory(self):
        inputs_path = os.path.join(self.root_directory, self.sources_directory)
        self.images_list = sorted([f for f in os.listdir(inputs_path) if TIFF_REGEX.match(f) is not None])
        if len(self.images_list) == 0: # Didn't find any file in the folder.
            show_info("Didn't find any TIFF file in the provided folder.")
            self.images_list = ['---']
        self.image_selector.clear()
        self.image_selector.addItems(self.images_list)
        return True
    
    def get_classes(self):
        classes = []
        for l in self.viewer.layers:
            if l.name.startswith(_CLASS_PREFIX):
                classes.append(l.name[len(_CLASS_PREFIX):])
        return classes
    
    def clear_classes_layers(self):
        names = [l.name for l in self.viewer.layers]
        for n in names:
            if n.startswith(_CLASS_PREFIX):
                self.viewer.layers[n].data = []

    def restore_classes_layers(self):
        classes_path = os.path.join(self.root_directory, "classes.txt")
        if not os.path.isfile(classes_path):
            show_info("No classes file found.")
            return
        classes = []
        with open(classes_path, "r") as f:
            classes = [item for item in f.read().split('\n') if len(item.strip()) > 0]
        for i, c in enumerate(classes):
            basis = c.strip()
            if len(basis) == 0:
                continue
            name = _CLASS_PREFIX + basis
            if name in self.viewer.layers:
                continue
            color = _COLORS[i % len(_COLORS)]
            self.viewer.add_shapes(
                name=name,
                edge_color=color,
                face_color="transparent",
                opacity=0.8,
                edge_width=3
            )
        show_info(f"Classes restored: {classes}")
    
    def add_labels(self, data):
        """
        Uses the dictionary of data to reset and refill shapes layers containing YOLO bounding-boxes.
        The class index refers to the index in which shape layers appear in the layers stack of Napari.
        """
        # Boxes are created according to the current image's size.
        h, w = self.viewer.layers[_IMAGE_LAYER].data.shape[:2]
        class_layers = [l.name for l in self.viewer.layers if l.name.startswith(_CLASS_PREFIX)]
        for c, bbox_list in data.items():
            rectangles = []
            for bbox in bbox_list:
                x1, y1, x2, y2 = self.yolo2bbox(bbox)
                xmin = int(x1*w)
                ymin = int(y1*h)
                xmax = int(x2*w)
                ymax = int(y2*h)
                points = np.array([[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]])
                rectangles.append(np.array(points))
            layer = self.viewer.layers[class_layers[c]]
            layer.data = rectangles
            layer.face_color='transparent'
            layer.edge_color=_COLORS[c % len(_COLORS)]
            layer.edge_width=3

    def load_annotations(self, labels_path):
        """
        Loads a file (.txt) containing annotations over the currently opened image.
        Parses the file and expects for each line:
        - (int) The box's class
        - (float) The box's x component
        - (float) The box's y component
        - (float) The box's width
        - (float) The box's height
        All these values are separated with spaces.
        The internal structure (a dictionary) makes a list of boxes per class index:
            data[class_index] = [(x1, y1, w1, h1), (x2, y2, w2, h2)]

        Args:
            - labels_path (str): The absolute path to the ".txt" file.
        """
        lines = []
        with open(labels_path, "r") as f:
            lines = f.read().split('\n')
        if len(lines) == 0:
            return
        data = dict()
        for line in lines:
            if line == "":
                continue
            c, x, y, w, h = line.split(" ")
            c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)
            data.setdefault(c, []).append((x, y, w, h))
        self.add_labels(data)

    def deselect_all(self):
        """
        Deselects everything in a shape layer.
        It is required when you flush the content of a shape layer when you open a new image.
        If you flush with an active selection, you will get an "index out of range" right away.
        Only targets shape layers representing YOLO classes.
        """
        for l in self.viewer.layers:
            if not l.name.startswith(_CLASS_PREFIX):
                continue
            l.mode = 'pan_zoom'
            l.selected_data = set()

    def open_image(self):
        """
        Uses the value contained in the `self.image_selector` to find and open an image.
        Reloads the annotations if some were already made for this image.
        """
        current_image = self.image_selector.currentText()
        # Check that the name is valid.
        if (self.root_directory is None) or (current_image is None) or (current_image == "---") or (current_image == ""):
            return
        image_path = os.path.join(self.root_directory, self.sources_directory, current_image)
        current_as_txt = TIFF_REGEX.match(current_image).group(1) + ".txt" # Remove extension + adds ".txt" extension.
        labels_path = os.path.join(self.root_directory, self.annotations_directory, current_as_txt)
        if not os.path.isfile(image_path):
            print(f"The image: '{current_image}' doesn't exist.")
            return
        data = tifffile.imread(image_path)
        if _IMAGE_LAYER in self.viewer.layers:
            self.viewer.layers[_IMAGE_LAYER].data = data
        else:
            self.viewer.add_image(data, name=_IMAGE_LAYER)
        self.deselect_all()
        self.restore_classes_layers()
        self.clear_classes_layers()
        if os.path.isfile(labels_path): # If some annotations already exist for this image.
            self.load_annotations(labels_path)
        self.count_boxes()

    def count_boxes(self):
        """
        Counts the number of boxes in each class to make sure annotations are balanced.
        """
        classes = self.get_classes()
        annotations_path = os.path.join(self.root_directory, self.annotations_directory)
        counts = dict()
        for f in os.listdir(annotations_path):
            if not f.endswith(".txt"):
                continue
            with open(os.path.join(annotations_path, f), "r") as file:
                lines = file.read().split('\n')
            for line in lines:
                if line == "":
                    continue
                c, _, _, _, _ = line.split(" ")
                c = int(c)
                counts[c] = counts.get(c, 0) + 1
        
        text = ""
        for class_idx, class_name in enumerate(classes):
            if class_idx not in counts:
                text += f'<span style="color: {_COLORS[class_idx % len(_COLORS)]}"><b>{class_name}</b></span>: 0<br>'
            else:
                text += f'<span style="color: {_COLORS[class_idx % len(_COLORS)]}"><b>{classes[class_idx]}</b></span>: {counts[class_idx]}<br>'

        self.count_display_label.setText(text)
        self.count_display_label.setMaximumHeight(50 * len(counts))