# Evaluating the Firefighter Optimization Algorithm for CNN Hyperparameter Tuning

This repository contains the full experimental pipeline for evaluating the **Firefighter Optimization Algorithm (FOA)** for **hyperparameter tuning of a CNN (VGG16)** applied to **brain tumor MRI classification**, and for comparing its performance against **Particle Swarm Optimization (PSO)** under strictly controlled and reproducible conditions.

The project was developed as part of a Bachelor’s thesis in **Applied Computer Science and Artificial Intelligence** and focuses on **metaheuristic optimization** in deep learning.

---

## Project Overview

- **Task:** Multi-class brain tumor MRI classification  
- **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor  
- **Backbone:** VGG16 (ImageNet-pretrained)  
- **Optimizers Compared:**  
  - Firefighter Optimization Algorithm (FOA)  
  - Particle Swarm Optimization (PSO)  
- **Objective:** Maximize validation accuracy under a fixed function-evaluation budget  
- **Frameworks:** Python, TensorFlow / Keras, NumPy  

---

## Dataset

The experiments use the **Kaggle Brain Tumor MRI Dataset**, consisting of **3,264 T1-weighted MRI slices** distributed across four classes.

Two dataset splitting strategies are supported:

1. **Kaggle official split** (Training / Testing folders)  
2. **Reproducible stratified 70/15/15 split** (train / val / test)  

All splits are **seeded and logged** to guarantee reproducibility.


---


## Hyperparameter Search Space

| Hyperparameter | Range |
|---------------|------|
| Dense units | 128 – 512 |
| Dropout rate | 0.25 – 0.55 |
| Learning rate | 1e-5 – 1e-4 (log-scale) |
| Batch size | {16, 32} |
| L2 weight decay | 1e-6 – 1e-4 |
| Unfrozen conv blocks | {0, 1, 2} |

Epochs are fixed during optimization to ensure fair comparison.

---

## Results

Each run stores:
- Best hyperparameters
- Validation accuracy curves
- Configuration files
- Logs for reproducibility

The final model is retrained on **train + validation** and evaluated on the **test set** using accuracy, precision, recall, F1-score, and AUC.

---

## Reproducibility

- All seeds logged
- Identical evaluation budgets
- Centralized evaluation wrapper

