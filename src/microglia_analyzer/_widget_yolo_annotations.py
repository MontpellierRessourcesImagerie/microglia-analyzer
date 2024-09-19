from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLineEdit,
                            QHBoxLayout, QPushButton, QLabel,
                            QFileDialog, QComboBox)

from napari.utils.notifications import show_info

import tifffile
import numpy as np
import os

_CLASS_PREFIX = "class."
_IMAGE_LAYER  = "Image"
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


class AnnotateBoundingBoxesWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        # Active Napari viewer.
        self.viewer = napari_viewer
        # Folder in which are located the 'images' and 'annotations' folders.
        self.sources_directory = None
        # List of images ('.tif') in the 'images' folder.
        self.images_list = []

        self.layout = QVBoxLayout()
        self.init_ui()
        self.setLayout(self.layout)

    def init_ui(self):
        # Label + text box for the inputs sub-folder's name:
        self.inputs_name_label = QLabel("Inputs sub-folder name:")
        self.inputs_name = QLineEdit()
        self.inputs_name.setText("inputs")
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.inputs_name_label)
        h_layout.addWidget(self.inputs_name)
        self.layout.addLayout(h_layout)

        # Label + text box for the annotations sub-folder's name:
        self.annotations_name_label = QLabel("Annotations sub-folder name:")
        self.annotations_name = QLineEdit()
        self.annotations_name.setText("labels")
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.annotations_name_label)
        h_layout.addWidget(self.annotations_name)
        self.layout.addLayout(h_layout)

        # Label + button to select the source directory:
        self.select_sources_directory_button = QPushButton("📂 Sources directory")
        self.select_sources_directory_button.clicked.connect(self.select_sources_directory)
        self.layout.addWidget(self.select_sources_directory_button)

        # Label + combobox containing inputs list:
        self.image_selector = QComboBox()
        self.image_selector.currentIndexChanged.connect(self.open_image)
        self.image_selector.addItem("---")
        self.layout.addWidget(self.image_selector)

        # Button to add a template layer:
        self.add_template_button = QPushButton("🔖 Add class layer")
        self.add_template_button.clicked.connect(self.add_template)
        self.layout.addWidget(self.add_template_button)

        # New name for the class layer + 'apply to current layer' button:
        self.new_name_label = QLabel("New class name:")
        self.new_name = QLineEdit()
        h_laytout = QHBoxLayout()
        h_laytout.addWidget(self.new_name_label)
        h_laytout.addWidget(self.new_name)
        self.apply_to_current_button = QPushButton("🎯 Rename class")
        self.apply_to_current_button.clicked.connect(self.apply_to_current)
        self.layout.addLayout(h_laytout)
        self.layout.addWidget(self.apply_to_current_button)

        # Button to save the annotations:
        self.save_button = QPushButton("💾 Save annotations")
        self.save_button.clicked.connect(self.save_state)
        self.layout.addWidget(self.save_button)
    
    def apply_to_current(self):
        name_candidate = self.new_name.text().lower().replace(" ", "-")
        if name_candidate == "":
            show_info("Empty name.")
            return
        if name_candidate.startswith(_CLASS_PREFIX):
            full_name = name_candidate
        else:
            full_name = _CLASS_PREFIX + name_candidate
        l = self.viewer.layers.selection.active
        if (l is not None) and ('face_color' in dir(l)):
            l.name = full_name
        self.new_name.setText("")

    def select_sources_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select sources directory")
        if not os.path.isdir(directory):
            show_info("Invalid directory.")
            return
        if not self.set_sources_directory(directory):
            show_info("No input directory found.")
            return

    def yolo2bbox(self, bboxes):
        xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
        xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
        return xmin, ymin, xmax, ymax
    
    def bbox2yolo(self, bbox):
        ymax, xmax = bbox[2]
        ymin, xmin = bbox[0]
        shape = self.viewer.layers[_IMAGE_LAYER].data.shape
        if len(shape) == 3:
            shape = shape[0:2]
        height, width = shape
        x = (xmin + xmax) / 2 / width
        y = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        return round(x, 3), round(y, 3), round(w, 3), round(h, 3)
    
    def points2yolo(self, layer_name, index):
        data = self.viewer.layers[layer_name].data
        tuples = []
        for rectangle in data:
            bbox = (index,) + self.bbox2yolo(rectangle)
            tuples.append(bbox)
        return tuples
    
    def write_annotations(self, tuples):
        labels_folder = os.path.join(self.sources_directory, self.annotations_name.text())
        labels_path = os.path.join(labels_folder, self.image_selector.currentText().replace(".tif", ".txt"))
        with open(labels_path, "w") as f:
            for row in tuples:
                f.write(" ".join(map(str, row)) + "\n")
        with open(os.path.join(self.sources_directory, "classes.txt"), "w") as f:
            for c in self.get_classes():
                f.write(c + "\n")
        show_info("Annotations saved.")
    
    def save_state(self):
        count = 0
        lines = []
        for l in self.viewer.layers:
            if not l.name.startswith(_CLASS_PREFIX):
                continue
            lines += self.points2yolo(l.name, count)
            count += 1
        self.write_annotations(lines)

    def add_template(self):
        if _IMAGE_LAYER not in self.viewer.layers:
            show_info("No image loaded.")
            return
        n_classes = len(self.get_classes())
        class_name = _CLASS_PREFIX + f"template-{str(n_classes+1).zfill(2)}"
        color = _COLORS[n_classes % len(_COLORS)]
        l = self.viewer.layers.selection.active
        if (l is not None) and (l.name.startswith(_CLASS_PREFIX)):
            l.selected_data = set()
        self.viewer.add_shapes(
            name=class_name,
            edge_color=color,
            face_color="#00000000",
            opacity=0.8,
            edge_width=3
        )

    def set_sources_directory(self, directory):
        inputs_path = os.path.join(directory, self.inputs_name.text())
        annotations_path = os.path.join(directory, self.annotations_name.text())
        if not os.path.isdir(inputs_path):
            return False
        if not os.path.isdir(annotations_path):
            os.makedirs(annotations_path)
        self.sources_directory = directory
        self.images_list = sorted([f for f in os.listdir(inputs_path)])
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
        if len(self.get_classes()) > 0:
            return
        classes_path = os.path.join(self.sources_directory, "classes.txt")
        if not os.path.isfile(classes_path):
            show_info("No classes file found.")
            return
        classes = []
        with open(classes_path, "r") as f:
            classes = f.read().split('\n')
        for i, c in enumerate(classes):
            if c == "":
                continue
            color = _COLORS[i % len(_COLORS)]
            self.viewer.add_shapes(
                name=_CLASS_PREFIX + c,
                edge_color=color,
                face_color="#00000000",
                opacity=0.8,
                edge_width=3
            )
        show_info(f"Classes restored: {classes}")
    
    def add_labels(self, data):
        shape = self.viewer.layers[_IMAGE_LAYER].data.shape
        if len(shape) == 3:
            shape = shape[0:2]
        h, w = shape
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
            layer.face_color='#00000000'
            layer.edge_color=_COLORS[c % len(_COLORS)]
            layer.edge_width=3

    def load_annotations(self, labels_path):
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
        for l in self.viewer.layers:
            if not l.name.startswith(_CLASS_PREFIX):
                continue
            l.mode = 'pan_zoom'
            l.selected_data = set()

    def open_image(self):
        current_image = self.image_selector.currentText()
        if (self.sources_directory is None) or (current_image is None) or (current_image == "---") or (current_image == ""):
            return
        image_path = os.path.join(self.sources_directory, self.inputs_name.text(), current_image)
        labels_path = os.path.join(self.sources_directory, self.annotations_name.text(), current_image.replace(".tif", ".txt"))
        if not os.path.isfile(image_path):
            return
        data = tifffile.imread(image_path)
        if _IMAGE_LAYER in self.viewer.layers:
            self.viewer.layers[_IMAGE_LAYER].data = data
        else:
            self.viewer.add_image(data, name=_IMAGE_LAYER)
        self.deselect_all()
        self.restore_classes_layers()
        self.clear_classes_layers()
        if os.path.isfile(labels_path):
            self.load_annotations(labels_path)

    def init_reset_state(self):
        """
        This function must:
            - Open the image currently pointed at by the combobox.
            - Clear all the classes layers.
        """
        self.clear_classes_layers()
        pass