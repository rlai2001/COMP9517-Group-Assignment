# COMP9517 Group Project – Aerial Image Classification (SkyView Dataset)

This repository presents a comprehensive evaluation of six aerial image classification models on the SkyView dataset. The project was completed for COMP9517: Computer Vision at UNSW, T1 2025. The models span from traditional methods to modern deep learning and transformer-based architectures.

## Project Overview

The task was to classify 12,000 aerial landscape images into 15 land-use categories using a range of models:
- Traditional Methods: SVM, K-NN, Random Forest (with SIFT, LBP features)
- Custom CNN: Lightweight CNN6-GAP (efficient for edge deployment)
- ResNet-18: Classic deep CNN with residual connections
- DenseNet-121: Feature reuse with densely connected layers
- Swin Transformer: Shifted window attention and hierarchical structure
- ViT-Base: Pure transformer-based image recognition using self-attention

All models were trained under a uniform training pipeline and evaluated with consistent augmentation, optimization strategies, and performance metrics.

## Experimental Setup

- Dataset: SkyView Aerial Landscape Dataset (12,000 images, 15 classes)  
- Image Size: Resized to 224x224  
- Training/Validation Split: 80/20  
- Augmentation: Random crop, flip, rotation, color jitter, noise, occlusion  
- Loss Functions: Cross-Entropy, Label Smoothing (where applicable)  
- Optimizers: Adam / AdamW  
- Schedulers: CosineAnnealingLR, Fixed LR  
- Metrics: Accuracy, Macro-F1, Loss

## Model Summary

| Model            | Accuracy (%) | Notes |
|------------------|--------------|-------|
| ViT-Base         | 99.58        | Highest global modelling performance |
| DenseNet-121     | 98.50        | Strong generalization, robust structure |
| ResNet-18        | 97.67        | Reliable baseline with residuals |
| Swin Transformer | 97.54        | Balanced local-global learning |
| CNN6-GAP         | 91.17        | Lightweight and edge-deployable |
| SVM (SIFT/LBP)   | 61.0         | Baseline traditional method |

## Robustness and Explainability

- Grad-CAM: Used to interpret model focus areas
- Robustness Testing:
  - Gaussian Noise
  - Blur
  - Occlusion

CNN6-GAP was most robust to noise and occlusion. DenseNet-121 exhibited performance degradation under blur due to limited robustness-specific training.

## Folder Structure

COMP9517-Group-Assignment/
├── CNN/          # Custom CNN6-GAP and training history  
├── Swin/         # Swin Transformer training and checkpoints  
├── DenseNet/     # DenseNet with GradCAM and robustness test  
├── ViT/          # ViT training pipeline  
├── ResNet/       # ResNet-18 implementation  
├── SVM/          # SVM, HOG, LBP, KNN, Random Forest  
├── Report.pdf   # Final report  
├── README.md  
├── requirements.txt  
└── .gitignore

## How to Run

Clone the repository and set up dependencies:

git clone https://github.com/rlai2001/COMP9517-Group-Assignment.git
cd COMP9517-Group-Assignment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

pip install -r requirements.txt

Each subfolder contains its own train.py, evaluation script, or notebook. Example:

python CNN/CNN_Train.py  
python DenseNet/evaluate_reportDenseNet.py  
python ViT/train_vit_skyview.py

## Contributors

| Name         | zID        | Contribution                        |
|--------------|------------|-------------------------------------|
| Shixun Li    | z5505146   | CNN, Swin Transformer               |
| Xinbo Li     | z5496624   | DenseNet, Grad-CAM                  |
| Richard Lai  | z5620374   | ResNet-18                           |
| Qiyun Li     | z5504759   | Vision Transformer                  |
| Junle Zhao   | z5447039   | SVM, LBP, k-NN, Random Forest       |

## License

This repository is intended for academic use only – COMP9517, T1 2025, UNSW Sydney.
