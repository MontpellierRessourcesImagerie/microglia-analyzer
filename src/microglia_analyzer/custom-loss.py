from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

def compute_skeleton_penalty(y_pred, min_length=10):
    # Binariser la prédiction
    y_pred_bin = y_pred > 0.5
    
    # Labeliser les composantes connexes
    labeled_image = label(y_pred_bin)
    
    penalty = 0
    for region in regionprops(labeled_image):
        # Créer un masque pour chaque région
        region_mask = labeled_image == region.label
        
        # Calculer le squelette de la région
        skeleton = skeletonize(region_mask)
        
        # Calculer la longueur du squelette
        skeleton_length = skeleton.sum()
        
        # Ajouter une pénalité si la longueur du squelette est inférieure au seuil
        if skeleton_length < min_length:
            penalty += (min_length - skeleton_length)
    
    return penalty

def custom_loss(y_true, y_pred):
    # Calcul de ta loss actuelle (par exemple, Dice Loss)
    dice_loss = dice_loss_function(y_true, y_pred)
    
    # Calcul des pénalités basées sur la longueur des skeletons
    skeleton_penalty = compute_skeleton_penalty(y_pred)
    
    # Combinaison des pénalités
    total_loss = dice_loss + gamma * skeleton_penalty
    
    return total_loss