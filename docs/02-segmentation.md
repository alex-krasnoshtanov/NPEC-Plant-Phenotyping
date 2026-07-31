# 2. Segmentation and Root Analysis

Roots are segmented with a U-Net, then analyzed geometrically.

- `train.py` - PyTorch training loop (segmentation-models-pytorch encoder), mixed
  precision, checkpoint caching. Writes `best_model.pth` (not committed; too large).
- `evaluate.py` - metrics on a held-out split.
- `check_setup.py` - environment/dependency sanity check.
- `root_analysis.py` - post-processing: keeps the N largest connected components above a
  minimum area (filtering noise before selecting, so tiny artifacts never crowd out real
  roots), separates plants by region of interest, and measures primary-root length.
- `training.ipynb`, `inference.ipynb` - the exploratory training and inference notebooks.
- `primary_root_lengths.csv` - a sample measurement output.

Note: an earlier TensorFlow/Keras variant of the U-Net also exists in notebook form; the
PyTorch path above is the canonical one used by the integrated system.
