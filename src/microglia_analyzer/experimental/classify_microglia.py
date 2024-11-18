import torch
import cv2
import numpy as np
import tifffile
import os
from microglia_analyzer.tiles.tiler import ImageTiler2D, normalize

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1: list [x1, y1, x2, y2], coordinates of the first box.
    - box2: list [x1, y1, x2, y2], coordinates of the second box.

    Returns:
    - iou: float, Intersection over Union
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

class MicrogliaClassifier(object):
    def __init__(self, model_path, image_path, iou_tr=0.8, score_tr=0.5, reload_yolo=False):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        if not model_path.endswith(".pt"):
            raise ValueError("Model file must be a '.pt' file")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=reload_yolo)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        self.image = normalize(tifffile.imread(image_path), 0, 255, np.uint8)
        self.bboxes = None
        self.iou_threshold = iou_tr
        self.score_threshold = score_tr
        self.classes = self.model.names

    def remove_useless_boxes(self, boxes):
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

        for i, (box1, score1, class1) in enumerate(zip(boxes['boxes'], boxes['scores'], boxes['classes'])):
            if i in used:
                continue
            chosen_box = box1
            chosen_score = score1
            chosen_class = class1
            for j, (box2, score2, class2) in enumerate(zip(boxes['boxes'], boxes['scores'], boxes['classes'])):
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
        return clean_boxes

    def inference(self):
        results = self.model(self.image)
        for img_results in results.xyxy:
            boxes   = img_results[:, :4].tolist()
            scores  = img_results[:, 4].tolist()
            classes = img_results[:, 5].tolist()
            self.bboxes = {
                'boxes'  : boxes,
                'scores' : scores,
                'classes': classes,
            }
    
    def get_cleaned_bboxes(self):
        return self.remove_useless_boxes(self.bboxes)

# -----------------------------------------------------------------

def draw_bounding_boxes(image, predictions, classes, exclude_class=1, thickness=2):
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
    box_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 255, 255), (0, 0, 0), (128, 128, 128), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (128, 128, 128)]
    image_with_boxes = image.copy()
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_GRAY2BGR)
    
    for box, cls, score in zip(predictions['boxes'], predictions['classes'], predictions['scores']):
        if int(cls) == exclude_class:
            continue
        x1, y1, x2, y2 = map(int, box)  # Cast into integers
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color=box_colors[int(cls)], thickness=thickness)
        label = f"{classes[int(cls)]} ({score:.2f})"
        cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[int(cls)], 1)
    
    return image_with_boxes


if __name__ == "__main__":
    mc = MicrogliaClassifier(
        "/home/benedetti/Documents/projects/2060-microglia/µyolo/µyolo-V051/weights/best.pt",
        "/home/benedetti/Documents/projects/2060-microglia/data/raw-data/tiff-data/adulte 3.tif"
    )
    mc.inference()
    cleaned_bboxes = mc.get_cleaned_bboxes()
    visual = draw_bounding_boxes(mc.image, mc.get_cleaned_bboxes(), mc.classes)
    cv2.imwrite("/tmp/visual.png", visual)