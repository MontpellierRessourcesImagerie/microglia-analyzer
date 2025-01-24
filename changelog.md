# Change Log

## Version 1.1.6 (24 Jan. 2025)

- Modified the output so it can be re-opened from the viewer.
- The classifications are stored in the YOLO format. Masks and skeletons are saved as TIFFs.
- There is still a "results.csv" file, but there is also an independant file per image.
- Re-opening an image will also re-load the associated settings.
- The documentation is in progress.
- Added the export of PNG for quick status check, that don't require Napari.
- Added the version number to the GUI for easier communication about issues.
- Added a checkbox to allow the user to remove automatically previous attempts.

## Version 1.1.5 (21 Jan. 2025)

- New YOLO version with a bigger model (`YOLOv5m` instead of `YOLOv5s`)
- The new training now locks the scale and the intensity shifts in the data augmentation.
- The index of the model's version is now written on the button.
- In developer's mode, it is now possible to **Shift+Click** on buttons to select local models (no downloading).

## Version 1.1.4 (19 Jan. 2025)

- Images normalization is now performed globally instead of locally to reduce the number of false positives.
- The classification model was updated from the version 051 to the version 077.
- The closing structuring element was changed from a diamong to a disk and its radius was reduced from 4 to 2.
- For the reason just above, the default probability threshold was lowered from 40% to 15%.
- The classification works now with a system of votes per label instead of taking every box above a certain score.
- It is now possible to toggle the visibility of the "Garbage" class.
