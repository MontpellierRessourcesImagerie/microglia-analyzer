
# Instanciate the model
model_path = "/home/benedetti/Documents/projects/2060-microglia/µyolo/µyolo-V049/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load image and split it into tiles
image_path = "/home/benedetti/Documents/projects/2060-microglia/data/raw-data/tiff-data/adulte 3.tif"
image = tifffile.imread(image_path)
image = normalize(image, 0, 255, np.uint8)

# Inference
results = model(image)

# Extract the results
output = []
for i, img_results in enumerate(results.xyxy):
    boxes   = img_results[:, :4].tolist() # Coordonnées des bounding boxes [x1, y1, x2, y2]
    scores  = img_results[:, 4].tolist()  # Scores de confiance
    classes = img_results[:, 5].tolist()  # Classes prédites (index des classes)
    
    # Ajouter les résultats à la collection Python
    output.append({
        'image_index': i,
        'boxes': boxes,
        'scores': scores,
        'classes': classes,
    })

def remove_useless_boxes(boxes, iou_threshold=0.8):
    """
    Fusions boxes with an IoU greater than `iou_threshold`.
    The box with the highest score is kept, whatever the two classes were.

    Parameters:
    - boxes: list of dict, chaque dict contient 'box' (coordonnées) et 'class'
    - iou_threshold: float, seuil d'IoU pour fusionner les boîtes

    Returns:
    - fused_boxes: list of dict, boîtes après fusion
    """
    fused_boxes = []
    used        = set()

    for i, box1 in enumerate(boxes):
        if i in used:
            continue
        chosen_box = box1['box']
        for j, box2 in enumerate(boxes):
            if j <= i or j in used:
                continue
            iou = calculate_iou(chosen_box, box2['box'])
            if iou > iou_threshold:
                chosen_box = chosen_box if box1['score'] > box2['score'] else box2['box']
                used.add(j)
        fused_boxes.append({'box': chosen_box, 'class': box1['class'], 'score': box1['score']})
        used.add(i)

    return fused_boxes