import tifffile
import os
from microglia_analyzer.tiles.tiler import ImageTiler2D
from microglia_analyzer.utils import normalize_batch
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
# os.environ['CUDA_VISIBLE_DEVICES']  = '-1'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['OMP_NUM_THREADS']       = '1'

class UNetWorker(object):

    def __init__(self, owner, model_path, image, tile_size=512, overlap=128):
        import tensorflow as tf
        from microglia_analyzer.dl.losses import dice_skeleton_loss, bce_dice_loss
        # Importance of the skeleton in the loss function.
        self.unet_skeleton_coef = 0.2
        # Importance of the BCE in the BCE-dice loss function.
        self.unet_bce_coef = 0.7
        self.segmentation_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "bcl": bce_dice_loss(self.unet_bce_coef),
                "dsl": dice_skeleton_loss(self.unet_skeleton_coef, self.unet_bce_coef)
            }
        )
        self.image = image
        self.tile_size = tile_size
        self.overlap = overlap
        self.mask = None
        self.owner = owner
    
    def inference(self):
        shape = self.image.shape
        tiles_manager = ImageTiler2D(self.tile_size, self.overlap, shape)
        tiles = np.array(tiles_manager.image_to_tiles(self.image))
        predictions = np.squeeze(self.segmentation_model.predict(tiles, batch_size=8))
        normalize_batch(predictions)
        probabilities = tiles_manager.tiles_to_image(predictions)
        self.owner.probability_map = probabilities