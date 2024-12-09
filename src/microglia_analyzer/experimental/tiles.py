import numpy as np
from PIL import Image
from microglia_analyzer.tiles.tiler import ImageTiler2D
import tifffile

def generate_checkerboard(width, height, num_squares_x, num_squares_y):
    """
    Génère une image de damier pour la vérification de l'UV mapping.

    Parameters:
    ----------
    width : int
        Largeur de l'image en pixels.
    height : int
        Hauteur de l'image en pixels.
    num_squares_x : int
        Nombre de cases horizontalement.
    num_squares_y : int
        Nombre de cases verticalement.
    colors : tuple of tuples, optional
        Couleurs des cases sous forme de tuples RGB. Par défaut, noir et blanc.

    Returns:
    -------
    Image.Image
        Image du damier générée.
    """
    # Calculer la taille de chaque case
    square_width = width // num_squares_x
    square_height = height // num_squares_y

    # Créer un tableau vide pour l'image
    checkerboard = np.zeros((height, width), dtype=np.uint8)

    for y in range(num_squares_y):
        for x in range(num_squares_x):
            # Déterminer la couleur de la case actuelle
            if (x + y) % 2 == 0:
                color = 0
            else:
                color = np.random.randint(0, 256, size=1)
            
            # Définir les coordonnées de la case
            x_start = x * square_width
            y_start = y * square_height
            x_end = x_start + square_width
            y_end = y_start + square_height

            # Remplir la case avec la couleur choisie
            checkerboard[y_start:y_end, x_start:x_end] = color

    # Convertir le tableau NumPy en image Pillow
    img = Image.fromarray(checkerboard)
    return img

if __name__ == "__main__":
    import os
    import tifffile
    import numpy as np
    output_path = "/tmp/dump/"
    os.makedirs(output_path, exist_ok=True)

    shapes = [
        (2048, 2048), 
        (1024, 1024)
    ]
    for shape in shapes:
        print("-----------")
        image = np.ones(shape, dtype=np.float32)
        tiles_manager = ImageTiler2D(512, 128, shape)
        for t in tiles_manager.layout:
            print(t)
        tiles = tiles_manager.image_to_tiles(image)
        merged = tiles_manager.tiles_to_image(tiles)
        for i in range(len(tiles)):
            tifffile.imwrite(os.path.join(output_path, f"{shape[0]}_{str(i).zfill(2)}.tif"), tiles_manager.blending_coefs[i])
        tifffile.imwrite(os.path.join(output_path, f"{shape[0]}_merged.tif"), merged)

if __name__ == "":
    # Générer une image de 2048x2048 avec des cases de 128x128
    checkerboard_img = np.squeeze(np.array(generate_checkerboard(2048, 2048, 16, 16)))
    tifffile.imwrite("/tmp/original.tif", checkerboard_img)
    tiles_manager = ImageTiler2D(512, 128, checkerboard_img.shape)
    tiles = tiles_manager.image_to_tiles(checkerboard_img)
    tifffile.imwrite("/tmp/checkerboard.tif", tiles)
    merged = tiles_manager.tiles_to_image(tiles)
    tifffile.imwrite("/tmp/merged.tif", merged)
    tifffile.imwrite("/tmp/coefs.tif", tiles_manager.blending_coefs)
    tifffile.imwrite("/tmp/gradient.tif", tiles_manager.tiles_to_image(tiles_manager.blending_coefs))