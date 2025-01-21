import torch
import cv2
import numpy as np
import tifffile
import os
import shutil
import warnings
from microglia_analyzer.tiles.tiler import ImageTiler2D, normalize

_BOX_COLORS = [
    (255,   0,   0), 
    (  0, 255,   0), 
    (  0,   0, 255), 
    (  0, 255, 255), 
    (255,   0, 255), 
    (  0, 255, 255),
    (255, 255, 255)
]

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
    # Model's path, data root, main name, output root
    def __init__(self, model_path, data_path, main_name, output_root):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        if not model_path.endswith(".pt"):
            raise ValueError("Model file must be a '.pt' file")
        if not os.path.isdir(output_root):
            raise FileNotFoundError(f"Output root '{output_root}' not found")
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Root directory {data_path} not found")
        self.root_path = data_path
        self.main_name = main_name
        self.image_path = os.path.join(data_path, main_name)
        if not os.path.isdir(self.image_path):
            raise FileNotFoundError(f"Images directory {self.image_path} not found")

        self.images_pool = [i.replace('.tif', '') for i in os.listdir(self.image_path) if i.endswith(".tif")]
        self.set_compare(os.path.join(data_path, main_name+"-labels"))
        
        warnings.simplefilter("ignore", FutureWarning)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.classes = self.model.names

        self.output_path = os.path.join(output_root, model_path.split(os.sep)[-3])
        self.image       = None
        self.image_name  = None

        if os.path.isdir(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

        self.iou_threshold   = 1.0
        self.score_threshold = 0.3
        self.bboxes          = None

        # Dictionary of GT available for comparison
        self.compare_gt = {}
        self.load_gt()
        print(f"Found {len(self.images_pool)} images to process.")

    def set_compare(self, compare_path):
        """
        If we need to compare the results after the inference with the ground truth.
        """
        if not os.path.isdir(compare_path):
            return
        available_labels = set([i.replace('.txt', '') for i in os.listdir(compare_path) if i.endswith(".txt")])
        self.images_pool = sorted(list(set(self.images_pool).intersection(available_labels)))
    
    def load_gt(self):
        """
        Load the ground truth boxes for each image.
        """
        labels_path = os.path.join(self.root_path, self.main_name+"-labels")
        for image_name in self.images_pool:
            img = tifffile.imread(os.path.join(self.image_path, image_name+".tif"))
            s = img.shape
            with open(os.path.join(labels_path, image_name+".txt"), "r") as f:
                lines = f.readlines()
            boxes = []
            classes = []
            for line in lines:
                class_id, x, y, w, h = map(float, line.split())
                x1 = int((x - w/2) * s[1])
                y1 = int((y - h/2) * s[0])
                x2 = int((x + w/2) * s[1])
                y2 = int((y + h/2) * s[0])
                boxes.append([x1, y1, x2, y2])
                classes.append(int(class_id))
            self.compare_gt[image_name] = {
                'boxes'  : boxes,
                'classes': classes
            }

    def next_image(self):
        # Returns True if an image was successfully loaded, False otherwise
        if len(self.images_pool) == 0:
            self.image_name = None
            self.image = None
            print("=== No image left. ===")
            return False
        self.image_name = self.images_pool.pop(0)+".tif"
        self.image = tifffile.imread(os.path.join(self.image_path, self.image_name))
        print(f"=== Processing ({str(len(self.images_pool)+1).zfill(2)}) {self.image_name} ===")
        return True

    def remove_useless_boxes(self, boxes, remove_class=0):
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
            if int(class1) == remove_class:
                continue
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

    def get_cleaned_bboxes(self, remove_class=-1):
        return self.remove_useless_boxes(self.bboxes, remove_class)

    def compare_with_gt(self, cleaned_bboxes):
        if os.path.splitext(self.image_name)[0] not in self.compare_gt:
            return
        gt = self.compare_gt[os.path.splitext(self.image_name)[0]]
        count = 0
        gt_boxes = gt['boxes']
        for i, box in enumerate(cleaned_bboxes['boxes']):
            for j, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(box, gt_box)
                if iou > 0.2:
                    count += 1
                    print((i, j), box, gt_box, iou)
                    break
        print(f"Found {count} boxes out of {len(gt_boxes)} in {self.image_name}.")

    def inference(self):
        img = normalize(self.image.copy(), 0, 255, np.uint8)
        tiles_manager = ImageTiler2D(512, 256, img.shape)
        tiles = tiles_manager.image_to_tiles(img, False)
        results = self.model(tiles)
        self.bboxes = {
            'boxes'  : [],
            'scores' : [],
            'classes': [],
        }
        for i, img_results in enumerate(results.xyxy):
            y, x = tiles_manager.layout[i].ul_corner
            boxes   = img_results[:, :4].tolist()
            boxes   = [[int(box[0] + x), int(box[1] + y), int(box[2] + x), int(box[3] + y)] for box in boxes]
            scores  = img_results[:, 4].tolist()
            classes = img_results[:, 5].tolist()
            self.bboxes['boxes'] += boxes
            self.bboxes['scores'] += scores
            self.bboxes['classes'] += classes
    
    def run_prediction(self):
        while self.next_image():
            self.inference()
            cleaned_bboxes = self.get_cleaned_bboxes()
            self.compare_with_gt(cleaned_bboxes)
            visual = draw_bounding_boxes(self.image, cleaned_bboxes, self.classes)
            cv2.imwrite(os.path.join(self.output_path, self.image_name), visual)

# -----------------------------------------------------------------

def draw_bounding_boxes(image, predictions, classes, exclude_class=0, thickness=2):
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
    image_with_boxes = normalize(image, 0, 255, np.uint8)
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_GRAY2BGR)
    
    for box, cls, score in zip(predictions['boxes'], predictions['classes'], predictions['scores']):
        if int(cls) == exclude_class:
            continue
        x1, y1, x2, y2 = map(int, box)  # Cast into integers
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color=_BOX_COLORS[int(cls)], thickness=thickness)
        label = f"{classes[int(cls)]} ({score:.2f})"
        cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLORS[int(cls)], 1)
    
    return image_with_boxes


def main():
    indices = [98,99]
    for i in indices:
        print("####   VERSION: ", i, "   ####")
        mc = MicrogliaClassifier(
            f"/home/benedetti/Documents/projects/2060-microglia/µyolo/µyolo-V{str(i).zfill(3)}/weights/best.pt",
            "/home/benedetti/Desktop/pour-test-2060/",
            "microglia",
            "/home/benedetti/Desktop/pour-test-2060/tests/"
        )
        mc.run_prediction()

if __name__ == "__main__":
    main()