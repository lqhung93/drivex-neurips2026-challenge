"""Column layout of the "mapcam" input (camera experiment): map encoder v1 columns, then K PCA components of the CLIP
embedding of the CAM_FRONT anchor image and K of the same image flipped left to right.

The image and flipped image columns are named img_left_k and img_right_k so that drivex26.mirror, which swaps
left and right twins, swaps the two views when it flips a row; the names say nothing about the image content.
"""

from drivex26.map_features import NAMES

K = 16
MAPCAM_NAMES = (*NAMES, *(f"img_left_{k}" for k in range(K)), *(f"img_right_{k}" for k in range(K)))
