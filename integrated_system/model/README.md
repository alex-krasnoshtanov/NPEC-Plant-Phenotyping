# Segmentation weights

The trained U-Net checkpoint `best_model.pth` (~279 MB) is intentionally NOT committed:
it exceeds GitHub's 100 MB per-file limit and is a build artifact, not source.

To obtain it:

- Train from scratch with `segmentation/train.py`, which writes `best_model.pth`, then
  copy it here, or
- Point the integrated system at a checkpoint elsewhere.

`dispensing_system.py` expects the file at `integrated_system/model/best_model.pth`.
