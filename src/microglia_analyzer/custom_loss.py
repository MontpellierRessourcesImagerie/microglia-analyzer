from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import numpy as np

def dice_loss_function(y_true, y_pred):
    # Implémentation classique de la Dice Loss
    smooth = 1.0
    intersection = np.sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def compute_connected_components_penalty(y_pred, max_allowed=1):
    # Labeliser les composantes connexes
    labeled_image = label(y_pred > 0.5)
    num_components = np.max(labeled_image)
    
    # Pénaliser si le nombre de composantes dépasse le seuil autorisé
    penalty = max(0, num_components - max_allowed)
    
    return penalty

def compute_skeleton_reward(y_pred, min_length=10):
    # Calculer le squelette des objets
    skeleton = skeletonize(y_pred > 0.5)
    
    # Calculer la longueur du squelette
    skeleton_length = np.sum(skeleton)
    
    # Récompenser si la longueur du squelette est au moins égale à la longueur minimale
    reward = max(0, skeleton_length - min_length)
    
    return reward

def custom_loss(y_true, y_pred, alpha=0.5, beta=0.3, gamma=0.2):
    # Calcul de la Dice Loss
    dice_loss = dice_loss_function(y_true, y_pred)
    
    # Pénalité pour les petites composantes connexes
    connected_components_penalty = compute_connected_components_penalty(y_pred)
    
    # Récompense pour la longueur du squelette
    skeleton_reward = compute_skeleton_reward(y_pred)
    
    # Combinaison des pertes et des récompenses
    total_loss = (alpha * dice_loss) + (beta * connected_components_penalty) - (gamma * skeleton_reward)
    
    return total_loss

