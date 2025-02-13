import cv2
import requests
import zipfile
import tempfile
import os
import numpy as np
import shutil
from contextlib import closing
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

def hex_to_bgr(color_hex):
    color_hex = color_hex.lstrip('#')
    if len(color_hex) == 8:
        r, g, b, a = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4, 6))
    elif len(color_hex) == 6:
        r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        a = 255
    else:
        raise ValueError(f"Invalid hex color: {color_hex}")

    return (b, g, r, a)

def convert_hex_colors_to_bgr():
    return np.array([hex_to_bgr(color) for color in BBOX_COLORS])

BBOX_COLORS_BGR = convert_hex_colors_to_bgr()

def save_as_fake_colors(images, bindings, class_names, output_path):
    image, mask = images
    if (len(image.shape) != 2) or (len(mask.shape) != 2) or (image.shape != mask.shape):
        raise ValueError("Incompatible inputs for visual check.")

    canvas = normalize(image, 0, 255, np.uint8)
    canvas = np.maximum(canvas, mask)
    canvas = draw_bounding_boxes(canvas, bindings, class_names)
    cv2.imwrite(output_path, canvas)


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
    import tifffile
    alpha_channel = np.zeros_like(image) + 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color_channels = np.zeros_like(image)
    for cls, (y1, x1, y2, x2) in bindings:
        if cls == exclude:
            continue
        alpha = BBOX_COLORS_BGR[cls][3]
        color = BBOX_COLORS_BGR[cls][:3]
        cv2.rectangle(color_channels, (x1, y1), (x2, y2), color=tuple([int(c) for c in color]), thickness=thickness)
        cv2.rectangle(alpha_channel , (x1, y1), (x2, y2), color=255-int(alpha)                    , thickness=thickness)
    alpha_channel  = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)
    alpha_channel  = alpha_channel.astype(np.float32) / 255.0
    color_channels = color_channels.astype(np.float32) / 255.0
    image          = image.astype(np.float32) / 255.0
    color_channels *= (1.0 - alpha_channel)
    image          *= (alpha_channel)
    line_height    = 45
    spacing        = 20
    font_scale     = 1.0
    height         = len(class_names) * (spacing + line_height)
    image = np.floor((image + color_channels)*255).astype(np.uint8)
    y, _, _ = image.shape
    image = cv2.copyMakeBorder(image, top=0, bottom=height, right=0, left=0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    r_h = 40
    r_w = 65
    for i, n in enumerate(class_names):
        s = (20, y+spacing+i*(spacing+line_height))
        e = (20+r_w, y+spacing+i*(spacing+line_height)+r_h)
        cv2.rectangle(
            image, 
            s,
            e,
            color=tuple([int(c) for c in BBOX_COLORS_BGR[i][:3]]),
            thickness=-1
        )
        cv2.putText(image, n, (20+r_w+10, y+int(spacing/2)+i*(spacing+line_height)+r_h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return image


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
            [y2, x1]   # Lower-left
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
            with closing(requests.get(url, stream=True, timeout=timeout)) as response:
                response.raise_for_status()
                with open(zip_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)

            if not zipfile.is_zipfile(zip_path):
                raise zipfile.BadZipFile("The downloaded file is not a valid ZIP archive.")
        
        except requests.exceptions.RequestException as e:
            print(f"Error while downloading the models: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during download: {e}")
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


def generate_random_bindings(image_size, num_bindings):
    bindings = []
    height, width = image_size
    for _ in range(num_bindings):
        cls = np.random.randint(0, 5)
        x1, x2 = sorted(np.random.randint(0, width, size=2))
        y1, y2 = sorted(np.random.randint(0, height, size=2))
        bindings.append((cls, (y1, x1, y2, x2)))
    return bindings

if __name__ == "__main__":
    import tifffile
    image = tifffile.imread("/home/benedetti/Desktop/microglia/P7- moelle spinale.tif")
    image = normalize(image, 0, 255, np.uint8)
    bindings = generate_random_bindings(image.shape, 20)
    class_names = ['garbage', 'amoeboid', 'rod', 'intermediate', 'homeostatic']
    r = draw_bounding_boxes(image, bindings, class_names)
    cv2.imwrite("/tmp/test.png", r)