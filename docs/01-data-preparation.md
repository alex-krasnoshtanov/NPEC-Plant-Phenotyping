# 1. Data Preparation

Raw NPEC plate images and their root masks are cleaned and cut into fixed-size patches
suitable for U-Net training.

- `preprocess.py` - normalizes images/masks and records an image-to-mask mapping.
- `check_masks.py` - validates that every image has a matching, non-empty mask.
- `patchify_pipeline.py` - pads each image so its dimensions divide evenly, then cuts
  256x256 patches at 50% overlap (`patchify`). Padding-before-patching guarantees loss-
  less reconstruction; the alternative (cropping) would discard border roots.
- `build_symlinks.py` / `build_symlinks_dataset24.py` - assemble train/val folders by
  linking rather than copying, to avoid duplicating a large dataset on disk.
- `data_preparation.ipynb` - walkthrough of the patching logic and reconstruction check.
- `patchify_experiments.ipynb` - quick visual sanity checks.

The raw dataset itself is not shipped (see the repo README).
