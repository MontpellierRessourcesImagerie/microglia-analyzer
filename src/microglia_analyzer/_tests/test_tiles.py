import pytest
import numpy as np
import itertools
from microglia_analyzer.tiles.tiler import ImageTiler2D, normalize
from microglia_analyzer.tiles.recalibrate import recalibrate_shape, recalibrate_image


""" TESTS:

    - [X] Test que des exceptions sont levées si les paramètres sont incorrects.
    - [X] Test que le nombre de tiles est correct sur chaque axe.
    - [X] Test que les tiles ont la même forme que ce qui est calculé.
    - [X] Test que le nombre de tiles correspond à la grid size.
    - [X] Test que le nombre de tiles correspond au nombre de coefficients de blending.
    - [X] Test que la normalization fonctionne correctement, avec les caps de valeurs.
    - [X] Test que les tiles fusionnées redonnent l'image d'origine.
    - [X] Test que l'assemblage des coefficients de blending donne un canvas à 1.0 (avec flat et gradient).
    - [X] Test que le découpage et l'assemblage fonctionne peu importe le nombre de channels.
    - [ ] Test que le recalibrage d'image fonctionne correctement.

"""

# (patch_size, overlap, (height, width))
_PATCHES_SETTINGS = [
    ( 64,   0, (  64,   64)),
    ( 64,   0, (  64,  128)),
    ( 64,   0, ( 128,   64)),
    ( 64,   0, ( 128,  128)),
    ( 64,  32, (  64,   64)),
    ( 64,  32, (  64,  128)),
    ( 64,  32, ( 128,   64)),
    ( 64,  32, ( 128,  128)),
    (512, 128, (2048, 2048)),
    (512, 128, (1024, 1024))
]

# (Number of patches Y-axis, Number of patches X-axis)
_GRID_GROUND_TRUTH = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 1),
    (1, 3),
    (3, 1),
    (3, 3),
    (5, 5),
    (3, 3)
]

# Blending between patches (for tiles_to_image)
_BLENDINGS = ['flat', 'gradient']

# Number of channels for the image
_N_CHANNELS = [1, 3, 5]

# Normalization targets: (target data type, lower bound, upper bound)
_NORMALIZE_BOUNDS = [
    (np.uint8  , 0  , 255),
    (np.uint16 , 0  , 65535),
    (np.float32, 0.0, 1.0),
    (np.float64, 0.0, 1.0),
    (np.float32, 0.2, 0.8)
]

# Random images to be normalized
_IMAGE_NORMALIZE = [
    np.random.randint(50, 200, (128, 128)).astype(np.uint8),
    np.random.randint(50, 200, (128, 128, 3)).astype(np.uint8),
    np.random.rand(128, 128).astype(np.float32),
    np.random.rand(128, 128, 3).astype(np.float64),
    np.random.randint(-1000, 1000, (128, 128)).astype(np.int16),
    (np.random.rand(128, 128, 3) * 500 - 250).astype(np.float32),
    np.random.randint(-128, 127, (64, 64)).astype(np.int8),
    np.random.randint(0, 65535, (512, 512, 3)).astype(np.uint16),
    np.ones((128, 128)).astype(np.float32),
    np.ones((128, 128, 3)).astype(np.float32),
    np.zeros((128, 128)).astype(np.float32),
    np.zeros((128, 128, 3)).astype(np.float32)
]

_SHAPES_RECALIBRATE = [
    (0.325, 'um', (100, 100), (100, 100)),
    (0.325, 'µm', (100, 100), (100, 100)),
    (0.5  , 'um', (100, 100), (65 , 65)),
    (0.125, 'um', (100, 100), (260, 260)),
    (500  , 'nm', (100, 100), (65 , 65)),
    (125  , 'nm', (100, 100), (260, 260))
]

# ---------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype, lower_bound, upper_bound, image", 
    list((*i, j) for (i, j) in itertools.product(_NORMALIZE_BOUNDS, _IMAGE_NORMALIZE))
)
def test_normalize(dtype, lower_bound, upper_bound, image):
    """
    Tests that for each pixel, the sum of the blending coefficients is equal to 1.
    To do so, we just check that cutting and remerging an image doesn't change it.
    Input images are randomly generated.
    """
    normalized_image = normalize(image, lower_bound, upper_bound, dtype)
    is_zero, is_unique_val, vals_match = False, False, False
    if np.abs(np.max(normalized_image)) < 1e-5 and np.abs(np.min(normalized_image)) < 1e-5:
        is_zero = True
    if np.max(normalized_image) - np.min(normalized_image) <= 1e-6:
        is_unique_val = True
    if (np.max(normalized_image) - upper_bound < 1e-5) and (np.min(normalized_image) - lower_bound < 1e-5):
        vals_match = True
    assert is_zero or is_unique_val or vals_match


@pytest.mark.parametrize("patch_size, overlap, shape, errType", [
    (64, 32, (128, 128, 128), ValueError),        # invalid image shape
    (64, 32, (128,)         , ValueError),        # invalid image shape
    (64, 0 , (32 , 32)      , ValueError),        # patch bigger than image
    (64, 32, (32 , 32)      , ValueError),        # patch bigger than image
    (64, 48, (128, 128)     , ValueError),        # overlap bigger than half patch size
    (0 , 0 , (128, 128)     , ZeroDivisionError), # patch size is 0
    (0 , 0 , (0  , 0)       , ZeroDivisionError)  # patch size is 0
])
def test_wrong_settings(patch_size, overlap, shape, errType):
    """
    Tests that wrong settings raise exceptions.
    """
    with pytest.raises(errType):
        _ = ImageTiler2D(patch_size, overlap, shape)


@pytest.mark.parametrize(
    "patch_size, overlap, shape, target", 
    [(*patch, grid) for patch, grid in zip(_PATCHES_SETTINGS, _GRID_GROUND_TRUTH)]
)
def test_grid_size(patch_size, overlap, shape, target):
    """
    Tests that the number of tiles on each axis is what we expect.
    """
    pe = ImageTiler2D(patch_size, overlap, shape)
    assert pe.get_grid_size() == target


@pytest.mark.parametrize(
    "patch_size, overlap, shape, target", 
    [(*patch, grid) for patch, grid in zip(_PATCHES_SETTINGS, _GRID_GROUND_TRUTH)]
)
def test_n_tiles(patch_size, overlap, shape, target):
    """
    Tests that the number of tiles matches what we expect from the grid size.
    Does the same thing for blending tiles. (There should be one blending tile per patch)
    """
    pe = ImageTiler2D(patch_size, overlap, shape)
    assert len(pe.get_layout()) == target[0] * target[1]
    assert len(pe.blending_coefs) == target[0] * target[1]


@pytest.mark.parametrize("patch_size, overlap, shape", _PATCHES_SETTINGS)
def test_theoritical_tiles_size(patch_size, overlap, shape):
    """
    Tests that all the theoritical tiles have the correct size.
    """
    pe = ImageTiler2D(patch_size, overlap, shape)
    for patch in pe.get_layout():
        assert patch.lr_corner[0] - patch.ul_corner[0] == patch_size
        assert patch.lr_corner[1] - patch.ul_corner[1] == patch_size


@pytest.mark.parametrize("patch_size, overlap, shape", _PATCHES_SETTINGS)
def test_actual_tiles_size(patch_size, overlap, shape):
    """
    Tests that all the tiles extracted from an image have the same size.
    That size must be equal to the patch size.
    """
    pe = ImageTiler2D(patch_size, overlap, shape)
    image = np.random.randint(0, 255, shape).astype(np.uint8)
    tiles = pe.image_to_tiles(image)
    for tile in tiles:
        assert tile.shape == (patch_size, patch_size)


@pytest.mark.parametrize(
    "patch_size, overlap, shape, blending, nChannels", 
    list((*i, j, k) for (i, j, k) in itertools.product(_PATCHES_SETTINGS, _BLENDINGS, _N_CHANNELS))
)
def test_blending_coefs_sum(patch_size, overlap, shape, blending, nChannels):
    """
    Tests that for each pixel, the sum of the blending coefficients is equal to 1.
    To do so, we just check that cutting and remerging the image doesn't change it.
    """
    pe = ImageTiler2D(patch_size, overlap, shape, blending)
    original_image = normalize(np.random.rand(*shape))
    if nChannels > 1:
        original_image = np.stack([original_image] * nChannels, axis=-1)
    tiles = pe.image_to_tiles(original_image)
    transformed_image = pe.tiles_to_image(tiles)
    diff = np.abs(original_image - transformed_image)
    assert np.max(diff) - np.min(diff) < 1e-5

@pytest.mark.parametrize("input_p_size, input_unit, input_shape, expected_shape", _SHAPES_RECALIBRATE)
def test_recalibrate_shape(input_p_size, input_unit, input_shape, expected_shape):
    """
    Tests that shapes are correctly recalibrated to match a pixel size of 0.325 µm.
    """
    new_shape = recalibrate_shape(input_shape, input_p_size, input_unit)
    assert new_shape == expected_shape
