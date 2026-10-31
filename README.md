# View-Flexible Keratoconus Detection with ConvNeXt

This repository contains a Kaggle notebook for classifying keratoconus from corneal maps. It trains a shared ConvNeXt encoder on multiple Pentacam maps, with map-type embeddings and attention-based pooling, so the same model can also evaluate four-view Orbscan data.

## Method

- Pentacam: seven corneal maps per eye for training, validation, ablations, and testing.
- Orbscan: four maps per eye used as an external validation set.
- Training augmentation: shared geometric transforms, view dropout, and device-style augmentation.
- Backbone comparison includes ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, DenseNet-121, and Swin-Tiny.
- Evaluation reports accuracy, sensitivity, specificity, F1, MCC, AUC, calibration, confidence intervals, ablations, and few-shot threshold calibration.

## Run

Open `keratoconus_detection_shared_encoder.ipynb` in Kaggle. The notebook expects datasets mounted at:

```text
/kaggle/input/datasets/nasiruddin66/keratoconus-datasets-2/Keratoconus datasets 2
```

The data should contain `Pentacam_Dataset/` and `ORBSCAN_Dataset/`. GPU execution is recommended.

Required packages include PyTorch, `timm`, pandas, NumPy, Pillow, matplotlib, and scikit-learn.

## Output

Results are written to `/kaggle/working/results/`, with figures, tables, predictions, and a final results archive in PNG/PDF and CSV formats.
