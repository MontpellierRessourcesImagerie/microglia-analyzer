import numpy as np

class Patch2D(object):

    def __init__(self, patch_size, overlap, indices, shape, grid):
        """
        Builds a representation of a patch in a 2D image.
        The patch is defined by its upper left and lower right corners, as well as its overlap with its neighbours.
        The overlap is defined as the number of pixels that are shared with the neighbour.
        Coordinates are in the Python order: (y, x), with Y being upside-down.

        Args:
            patch_size: (int) Size of the patch (height and width), overlap included.
            overlap: (int) Overlap between patches.
            indices: (tuple) Indices of the patch in the grid (vertical and horizontal index in the grid of patches).
            shape: (tuple) Shape of the image.
            grid: (tuple) Number of patches on each axis.
        """
        # Total height and width of patches, including overlap (necessarily a square).
        self.patch_size    = patch_size
        # Minimum overlap between patches, in number of pixels.
        self.overlap       = overlap
        # Indices (y, x) of this patch within the grid.
        self.indices       = indices
        # Shape of the global image.
        self.shape         = shape
        # Grid size (number of patches on each axis).
        self.grid          = grid
        # Step between each patch.
        self.step          = patch_size - overlap
        # Does the current patch have a neighbour on each side?
        self.has_neighbour = [False, False, False, False]
        # Overlap size with each neighbour.
        self.overlaps      = [0    , 0    , 0    , 0]
        # Upper left corner of the patch.
        self.ul_corner     = None
        # Lower right corner of the patch.
        self.lr_corner     = None
        # -----
        self._process_patch()
        self._check_neighbours()
        self._process_overlap()
    
    def __str__(self):
        return f"Patch2D({self.patch_size} > {self.ul_corner}, {self.lr_corner}, {self.overlaps})"

    def _process_patch(self):
        """
        From the indices on the grid, determines the upper-left and lower-right corners of the patch.
        If the patch is on an edge, the overlap is increased to conserve a constant patch size.
        The upper-left corner is processed from the lower-right corner.
        On both axes, the lower coordinate is included, while the upper one is excluded.
        It implies that the last patch will contain indices corresponding to the shape of the image.
        """
        height, width = self.shape
        y, x = self.indices[0] * self.step, self.indices[1] * self.step
        lower_right = (
            min(y + self.patch_size, height), 
            min(x + self.patch_size, width)
        )
        upper_left = (
            lower_right[0] - self.patch_size,
            lower_right[1] - self.patch_size
        )
        self.lr_corner = lower_right
        self.ul_corner = upper_left
    
    def _check_neighbours(self):
        """
        Determines the presence of neighbours for the current patch.
        For the left and top edges, we just hav eto check if we are touching the index 0.
        For the right and bottom edges, we have to check if the indices match the grid size.
        Note that if the overlap is set to 0, the patches won't ever have any neighbour.
        """
        y, x = self.indices
        self.has_neighbour[0] = (self.overlap > 0) and (x > 0)
        self.has_neighbour[1] = (self.overlap > 0) and (y < self.grid[0] - 1)
        self.has_neighbour[2] = (self.overlap > 0) and (x < self.grid[1] - 1)
        self.has_neighbour[3] = (self.overlap > 0) and (y > 0)

    def _process_overlap(self):
        """
        According to the presence of neighbours, determines the overlap size with each neighbour.
        The overlap size varies depending on the position of the patch in the image. If we are on an edge, the overlap is increased.
        In the case of the bottom and right edges, we also check whether the next patch would exceed the image size.
        Otherwise, the overlap wouldn't be symmetric.
        """
        y, x = self.indices[0] * self.step, self.indices[1] * self.step
        if self.has_neighbour[0]:
            self.overlaps[0] = max(self.overlap, x - self.ul_corner[1])
        if self.has_neighbour[1]:
            self.overlaps[1] = self.overlap if (self.lr_corner[0] + self.patch_size - self.overlap <= self.shape[0]) else (self.shape[0] - self.lr_corner[0])
        if self.has_neighbour[2]:
            self.overlaps[2] = self.overlap if (self.lr_corner[1] + self.patch_size - self.overlap <= self.shape[1]) else (self.shape[1] - self.lr_corner[1])
        if self.has_neighbour[3]:
            self.overlaps[3] = max(self.overlap, y - self.ul_corner[0])
    
    def as_napari_rectangle(self):
        """
        Returns this bounding-box as a Napari rectangle, which is a numpy array:
        [
            [yMin, xMin],
            [yMin, xMax],
            [xMax, yMin],
            [xMax, yMax]
        ]
        """
        return np.array([
            [self.ul_corner[0], self.ul_corner[1]],
            [self.ul_corner[0], self.lr_corner[1]],
            [self.lr_corner[0], self.lr_corner[1]],
            [self.lr_corner[0], self.ul_corner[1]]
        ])