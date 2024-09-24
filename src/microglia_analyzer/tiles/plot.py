import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_patches_layout(extractor):
    _, ax = plt.subplots()
    shape = extractor.shape
    canvas = np.zeros(shape, dtype=np.uint8)

    ax.imshow(canvas, cmap='gray')

    for i, patch in enumerate(extractor.get_layout()):
        top_left = patch.ul_corner
        bottom_right = patch.lr_corner
        y1, x1 = top_left
        y2, x2 = bottom_right

        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), 
            width, 
            height, 
            linewidth=3, 
            edgecolor='red', 
            facecolor='none',
            alpha=0.5
        )
        ax.add_patch(rect)

    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)

    plt.show()