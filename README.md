# Stanford CS231n — Assignment 1

[![CS231n](https://img.shields.io/badge/Stanford-CS231n-8B0000?style=flat-square&logo=stanford&logoColor=white)](http://cs231n.stanford.edu/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

> **Stanford CS231n: Convolutional Neural Networks for Visual Recognition**
> Instructor: Fei-Fei Li, Justin Johnson, Serena Yeung
> Completed: June 2026

## Overview

Assignment 1 builds foundational Computer Vision classifiers **from scratch using NumPy** — no deep learning frameworks allowed. This teaches the mathematical underpinnings of every modern CV system.

## What's Implemented

| Module | Technique | Key Concepts |
|--------|-----------|-------------|
| **kNN** | k-Nearest Neighbors | L1/L2 distance, cross-validation |
| **SVM** | Multiclass SVM | Hinge loss, regularization, SGD |
| **Softmax** | Softmax Classifier | Cross-entropy loss, CIFAR-10 |
| **Two-Layer Net** | Neural Network | Backpropagation, gradient descent |
| **Features** | Feature Extraction | HOG, color histogram, feature engineering |

## Results

- **SVM:** ~39% accuracy on CIFAR-10
- **Softmax:** ~39% accuracy on CIFAR-10
- **Two-Layer NN:** ~50% accuracy on CIFAR-10
- **With features:** ~55% accuracy on CIFAR-10

## Tech Stack

- **Language:** Python 3.8+
- **Core:** NumPy (vectorized implementations — no loops)
- **Visualization:** matplotlib
- **Environment:** Jupyter Notebooks

## Key Learnings

- Implemented gradient descent **manually** — understands exactly what PyTorch's `autograd` does under the hood
- Vectorized gradient computation for efficiency
- Cross-validation for hyperparameter tuning
- Difference between parametric (SVM, Softmax) and non-parametric (kNN) classifiers

## How to Run

```bash
pip install numpy matplotlib
jupyter notebook knn.ipynb   # Start here
```

## Course Context

CS231n is Stanford's premier Computer Vision course. This assignment was completed as part of a self-directed curriculum in ML/AI engineering with a Computer Vision specialization.
