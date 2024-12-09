import os
import tifffile
import numpy as np
from microglia_analyzer.tiles.tiler import ImageTiler2D, normalize
from microglia_analyzer.dl.losses import (dice_skeleton_loss, bce_dice_loss, dice_loss)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def normalize_batch(batch):
    for i in range(len(batch)):
        batch[i] = normalize(batch[i])
    

skeleton_coef = 0.2
bce_coef      = 0.7

class MicrogliaSegmenter(object):
    def __init__(self, model_path, image_path, tile_size=512, overlap=256):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        if not model_path.endswith(".keras"):
            raise ValueError("Model file must be a '.keras' file")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "bcl": bce_dice_loss(bce_coef),
                "_dice_skeleton_loss": dice_skeleton_loss(skeleton_coef, bce_coef)
            }
        )
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        self.image = tifffile.imread(image_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.mask = None
    
    def run(self):
        pred = self.inference()
        t = self.get_threshold_val(pred)
        self.mask = self.clean_mask(pred, t)
    
    def inference(self):
        shape = self.image.shape
        tiles_manager = ImageTiler2D(self.tile_size, self.overlap, shape)
        tiles = np.array(tiles_manager.image_to_tiles(self.image))
        predictions = np.squeeze(self.model.predict(tiles, batch_size=8))
        tifffile.imwrite("/tmp/tiles.tif", predictions)
        tifffile.imwrite("/tmp/coefs.tif", tiles_manager.blending_coefs)
        # normalize_batch(predictions)
        probabilities = tiles_manager.tiles_to_image(predictions)
        return probabilities


if __name__ == "__main__":
    output_path = "/home/benedetti/Downloads/training-audrey/output/"
    model_path  = "/home/benedetti/Downloads/training-audrey/models/unet-V007/best.keras"
    folder_path = "/home/benedetti/Downloads/training-audrey/raw/"
    content     = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
    for i, image_name in enumerate(content):
        print(f"{i+1}/{len(content)}: {image_name}")
        image_path = os.path.join(folder_path, image_name)
        ms = MicrogliaSegmenter(model_path, image_path)
        tifffile.imwrite(os.path.join(output_path, image_name), ms.inference())
    print("DONE.")


"""
WORKFLOW DEPUIS LA GUI:

    1.  L'utilisateur sélectionne le dossier dans lequel se trouvent les images.
    2.  Le menu déroulant se rempli avec le contenu du dossier, les résultats existants sont chargés.
    3.  L'utilisateur sélectionne une image.
    4.  Dans la section "Segmentation", l'utilisateur peut choisir le modèle à utiliser.
    5.  Il peut ensuite cliquer sur "Segmenter" pour lancer le processus.
    6.  Un champ de float gère le threshold pour la transformation en masque.
        Une dilation a lieu automatiquement après le seuillage.
    7.  L'utilisateur peut maintenant passer au panneau de classification.
    8.  Il peut y choisir le modèle et le seuil minimal de confiance.
    9.  Il peut ensuite cliquer sur "Classifier" pour lancer le processus.
    10. À ce moment là, l'utilisateur doit pouvoir éditer ses résultats.
    11. On peut maintenant passer à la section "Measure".
        Ici, l'utilisateur peut transformer son masque en squelette.
        Pour chaque instance, on a la longueur totale, le nombre de branches, le nombre de feuilles, ...
    12. Enfin, il y a un bouton pour exporter les résultats.
        Le CSV consiste en une ligne par instance, avec sa classe et ses mesures.
"""