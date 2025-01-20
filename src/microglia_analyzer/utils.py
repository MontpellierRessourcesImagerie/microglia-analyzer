import cv2
import requests
import zipfile
import tempfile
import os
import numpy as np
import shutil
from microglia_analyzer import TIFF_REGEX
from microglia_analyzer.tiles.tiler import normalize

"""
Colors used in the `_widget.py` file, in the `show_classification` method.
Each color corresponds to a class (Garbage, Amoeboid, Rod, Intermediate and Homeostatic).
The alpha is included in the color, but it only affects the transparency in the viewer, not in the widget.
Indeed, Qt (QColor) doesn't handle the fact that the alpha is included in the color.
"""
BBOX_COLORS = [
    '#FFFFFF55',  # White
    '#00FF00FF',  # Green
    '#FFFF00FF',  # Yellow 
    '#00FFFFFF',  # Cyan 
    '#FF0000FF'   # Red 
]


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    It is designed to work with the format [x1, y1, x2, y2] (Raw YOLO output).

    Args:
        - box1 (list or tuple): [x1, y1, x2, y2], coordinates of the first box.
        - box2 (list or tuple): [x1, y1, x2, y2], coordinates of the second box.

    Returns:
        (float): Intersection over Union, a value between 0 and 1.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def normalize_batch(batch):
    """
    Performs a basic normalization (histogram stretching) on a batch of images.
    The normalization is done in place.
    The new data is in the range [0.0, 1.0].
    """
    for i in range(len(batch)):
        batch[i] = normalize(batch[i])


def draw_bounding_boxes(image, bindings, class_names, exclude=-1, thickness=2):
    """
    Draw bounding-boxes on an image.
    There is the possibility to exclude a class by its index.
    The output image is in BGR format and the modification is not performed in-place.

    Parameters:
    - image (np.array): Canvas on which the bounding-boxes will be drawn.
    - bindings ([(int, (int, int, int, int))]): List of bindings, each containing a class and a bounding-box: (cls, (y1, x1, y2, x2)).
    - class_names ([str]): List of class names.
    - exclude (int): Class to be excluded.
    - thickness (int): Thickness of the bounding-boxes outlines (default=2).
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for cls, (y1, x1, y2, x2) in bindings:
        if cls == exclude:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color=BBOX_COLORS[cls], thickness=thickness)
        label = f"{class_names[int(cls)]}"
        cv2.putText(canvas, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLORS[cls], 1)
    return canvas


def bindings_as_napari_shapes(bindings, exclude=-1):
    """
    From the bindings, creates a list of rectangles and a list of colors following the format expected by Napari.
    Both lists have the same size.
    There is the possibility to exclude a class by its index.

    Args:
        - bindings ([(int, (int, int, int, int))]): List of bindings, each containing a class and a bounding-box: (cls, (y1, x1, y2, x2)).
        - exclude (int): Class to be excluded.
    
    Returns:
        (list, list): List of rectangles and list of colors.
    """
    items = []
    colors = []
    for cls, (y1, x1, y2, x2) in bindings:
        if cls == exclude:
            continue
        rect = np.array([
            [y1, x1],  # Upper-left
            [y1, x2],  # Upper-right
            [y2, x2],  # Lower-right
            [y2, x1],  # Lower-left
        ])
        colors.append(BBOX_COLORS[cls])
        items.append(rect)
    return items, colors


def download_from_web(url, extract_to, timeout=100):
    """
    This function is used to download and extract a ZIP file from the web.
    In this project, it is used to download the pre-trained models (for both the UNet and the YOLO).
    To get it working, the files must have been bundled at level-0.
    It means that when you create your archive, you must do it from inside the folder (Ctrl+A > Compress), not from the parent folder itself.

    Args:
        - url (str): URL of the ZIP file to download.
        - extract_to (str): Parent folder where the ZIP file will be extracted. 
                            A new folder will be created inside.
        - timeout (int): Maximum time to wait for the download (default=100).
    """
    if os.path.isdir(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "downloaded.zip")
        print(f"Downloading model from {url}...")
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(zip_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
        except requests.exceptions.RequestException as e:
            print(f"Error while downloading the models: {e}")
            raise

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Model extracted to: {extract_to}")
        except zipfile.BadZipFile as e:
            print(f"Error while decompressing the model's ZIP: {e}")
            raise
        except Exception as e:
            print(f"Unknown decompression error: {e}")
            raise

def get_all_tiff_files(folder_path, no_ext=False):
        """
        Probes a folder and filters its content with a regex.
        All the TIFF are returned, whatever the number of 'f' or the case.
        If the `no_ext` attribute is True, the name is returned without the extension.
        """
        tiff_files = []
        for file_name in os.listdir(folder_path):
            match = TIFF_REGEX.match(file_name)
            if match:
                if no_ext:
                    tiff_files.append(match.group(1))
                else:
                    tiff_files.append(match.group(0))
        return sorted(tiff_files)
