# Breast Cancer Classification - Neural Network from Scratch

## Project Overview

This project implements a simple **Artificial Neural Network (ANN)** from scratch using **NumPy** for binary classification of breast cancer data. The goal is to classify tumors as **malignant** (M) or **benign** (B) based on the features of cell nuclei extracted from images.

The dataset used is the **Wisconsin Breast Cancer Dataset**, which contains 30 feature columns, and the labels are binary (B/M). This neural network is trained to predict whether a tumor is malignant or benign.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Neural Network Architecture](#neural-network-architecture)
- [Training](#training)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)

---

## Dataset

The dataset used for training and testing the neural network is the **Wisconsin Breast Cancer Dataset**, which includes the following:
- 30 features representing different characteristics of the cell nuclei (e.g., radius, texture, perimeter, area, smoothness).
- Labels: `M` for malignant and `B` for benign tumors.

### Preprocessing:
- The input features are normalized to a range of [0,1] to ensure efficient training.
- The labels are converted from 'M' (Malignant) to `1` and 'B' (Benign) to `0`.

---

## Neural Network Architecture

The architecture of the neural network is flexible and can be changed as required. The structure used for this project is:
- **Input Layer:** 30 neurons (one for each feature).
- **Hidden Layers:** Configurable; typically uses 1-3 hidden layers with various neuron counts.
- **Output Layer:** 1 neuron (for binary classification).

### Activation Function:
- **Sigmoid Activation** is used in all layers to squash the values between 0 and 1.

### Hyperparameters:
- **Learning Rate:** Controls how much the weights are adjusted after each epoch.
- **Momentum Factor:** Helps accelerate convergence by considering past weight updates.
- **Epochs:** Number of iterations to run during training.
- **Batch Size:** Currently, the network is trained using a single data point at a time (stochastic gradient descent).

---

## Training

The network is trained using **backpropagation** with stochastic gradient descent. The following steps are involved:

1. **Forward Propagation:** Pass inputs through the network to calculate the output.
2. **Backward Propagation:** Compute gradients and update the weights based on the difference between predicted and actual outputs.
3. **Evaluation:** Accuracy and confusion matrix are used to evaluate the model performance.

### Confusion Matrix
The model reports the following values in the confusion matrix:
- **True Positive (TP)**
- **False Positive (FP)**
- **False Negative (FN)**
- **True Negative (TN)**

---

## Dependencies

This project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## How to Run

1. Clone the repository or download the code files.
2. Ensure that the dataset (`wisc_bc_data.csv`) is placed in the same directory as the script or provide the correct file path.
3. Adjust the network architecture, learning rate, momentum factor, and epochs as needed in the code.
4. Run the code:

```bash
python NeuralNetwork.py
```

5. The training progress, including accuracy per epoch and confusion matrix, will be printed in the terminal.

---

## Results

The following experiments were performed with varying hyperparameters:

| Momentum Factor | Learning Rate | Epochs | Layers        | Training Accuracy |
|-----------------|---------------|--------|---------------|-------------------|
| 0.9             | 0.01          | 10000  | [5, 4, 1]     | 99.56%            |
| 0.9             | 0.01          | 6000   | [6, 5, 4, 1]  | 99.78%            |
| 0.9             | 0.01          | 10000  | [6, 5, 4, 1]  | 99.78%            |
| 0.5             | 0.09          | 2000   | [5, 4, 1]     | 99.34%            |

*(See the detailed results in the attached [PDF](./Output_project1.pdf))*
