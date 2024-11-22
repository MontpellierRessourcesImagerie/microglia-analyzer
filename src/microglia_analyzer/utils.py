import cv2
import requests
import zipfile
import tempfile
import os
import numpy as np
import shutil
from microglia_analyzer.tiles.tiler import normalize

BBOX_COLORS = [
    '#FF000000',  # Red
    '#0000FF',  # Blue
    '#FFFF00',  # Yellow
    '#008000',  # Dark green
    '#FFA500',  # Orange
    '#00FFFF',  # Cyan
    '#800080',  # Purple
    '#FF1493',  # Vivid pink
    '#ADFF2F',  # Lime
]

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        - box1 (list): [x1, y1, x2, y2], coordinates of the first box.
        - box2 (list): [x1, y1, x2, y2], coordinates of the second box.

    Returns:
        (float): Intersection over Union
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return 0.0 if union_area == 0 else (inter_area / union_area)

def normalize_batch(batch):
    for i in range(len(batch)):
        batch[i] = normalize(batch[i])

def draw_bounding_boxes(image, predictions, classes, thickness=2):
    """
    Dessine les bounding boxes sur une image en excluant une classe spécifique.

    Parameters:
    - image: np.array, image d'entrée (modifiée en place)
    - predictions: list of dict, liste des prédictions contenant les clés 'box' et 'class'
      Exemple : [{'box': [x1, y1, x2, y2], 'score': 0.95, 'class': 2}, ...]
    - exclude_class: int, la classe à exclure (par défaut 1)
    - box_color: tuple, couleur des bounding boxes (par défaut vert)
    - thickness: int, épaisseur des bounding boxes (par défaut 2)

    Returns:
    - image_with_boxes: np.array, l'image avec les bounding boxes dessinées
    """
    image_with_boxes = image.copy()
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_GRAY2BGR)
    
    for box, cls, score in zip(predictions['boxes'], predictions['classes'], predictions['scores']):
        x1, y1, x2, y2 = map(int, box)  # Cast into integers
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color=BBOX_COLORS[int(cls)], thickness=thickness)
        label = f"{classes[int(cls)]} ({score:.2f})"
        cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLORS[int(cls)], 1)
    
    return image_with_boxes

def boxes_as_napari_shapes(collection, swap=False):
    items = []
    colors = []
    for (cls, _, seg_bbox) in collection:
        y1, x1, y2, x2 = map(int, seg_bbox)
        if swap:
            x1, y1, x2, y2 = map(int, seg_bbox)
        rect = np.array([
            [y1, x1],  # Haut-gauche
            [y1, x2],  # Haut-droite
            [y2, x2],  # Bas-droite
            [y2, x1],  # Bas-gauche
        ])
        colors.append(BBOX_COLORS[int(cls)])
        items.append(rect)
    return items, colors

def download_from_web(url, extract_to, timeout=100):
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
