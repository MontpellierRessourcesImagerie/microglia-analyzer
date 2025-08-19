import numpy as np

def make_checkboard(shape, size, type):
    if type not in ('grayscale', 'rgb'):
        raise ValueError("type must be either 'grayscale' or 'rgb'")

    if type == 'grayscale':
        if len(shape) != 2:
            raise ValueError("For grayscale, shape must be (Y, X)")
        height, width = shape
        img = np.zeros((height, width), dtype=np.uint16)

    elif type == 'rgb':
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError("For rgb, shape must be (3, Y, X)")
        _, height, width = shape
        img = np.zeros((3, height, width), dtype=np.uint8)

    n_y = (height + size - 1) // size
    n_x = (width + size - 1) // size

    board = (np.indices((n_y, n_x)).sum(axis=0) % 2)

    for y in range(n_y):
        for x in range(n_x):
            y0 = y * size
            x0 = x * size
            y1 = min((y + 1) * size, height)
            x1 = min((x + 1) * size, width)

            if board[y, x] == 1:
                if type == 'grayscale':
                    value = np.random.randint(0, 65536, dtype=np.uint16)
                    img[y0:y1, x0:x1] = value
                else:  # rgb
                    value = np.random.randint(0, 256, size=(3, 1, 1), dtype=np.uint8)
                    img[:, y0:y1, x0:x1] = value

    return img


if __name__ == "__main__":
    import tifffile

    # Damier grayscale 512x512, cases de 64x64
    check = make_checkboard((512, 517), 64, 'grayscale')

    # Damier RGB 3x512x512, cases de 32x32
    check_rgb = make_checkboard((3, 512, 1024), 32, 'rgb')

    # Save the images
    tifffile.imwrite('/tmp/checkboard_gray.tif', check)
    tifffile.imwrite('/tmp/checkboard_rgb.tif' , check_rgb)
