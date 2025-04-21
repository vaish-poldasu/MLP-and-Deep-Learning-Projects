# MLP-and-Deep-Learning-Projects - MLP, GMM, PCA, and Autoencoder Implementation

This assignment involves implementing several machine learning models from scratch, including Multi-Layer Perceptrons (MLPs) for different tasks, a Gaussian Mixture Model (GMM) for image segmentation, Principal Component Analysis (PCA) for dimensionality reduction, and an Autoencoder for anomaly detection.

## 1. Multi-Layer Perceptron (MLP)

### 1.1 MLP for Multi-Class Classification (q1_a)

In this task, a configurable MLP classifier needs to be implemented from scratch for classifying handwritten symbols from a dataset. The implementation should support different activation functions, optimizers, and allow customization of the number and size of hidden layers. It should use forward and backpropagation mechanisms for training. The model must be evaluated using 10-fold cross-validation, and the mean and standard deviation of the accuracy scores across folds should be reported. The goal is also to analyze how hyperparameter choices affect the model’s performance and reliability.

### 1.2 MLP Regressor for Housing Price Prediction (q1_b)

This task involves implementing an MLP-based regression model from scratch to predict housing prices in Bangalore. The dataset must be cleaned by handling missing values and outliers. After preprocessing, the data should be split into training, validation, and test sets, and normalized. The regressor must be trained using configurable hyperparameters such as learning rates, activation functions, and network architectures. The final model should be evaluated using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared. Additionally, training loss curves should be plotted, and the effect of different configurations on prediction performance must be discussed.

### 1.3 MLP for Multi-Label Classification (q1_c)

In this problem, an MLP model for multi-label classification needs to be implemented from scratch to classify news articles into multiple categories. The text data should be preprocessed, TF-IDF features should be extracted, and the labels should be binarized. The model must support configurable hyperparameters and implement forward and backpropagation. Its performance should be evaluated using accuracy and Hamming loss metrics. The assignment also requires examining the impact of different hyperparameters and plotting the training curves.

## 2. Gaussian Mixture Model (q2)

A Gaussian Mixture Model should be implemented from scratch and used to segment an MRI brain scan into gray matter, white matter, and cerebrospinal fluid. The segmentation performance must be evaluated by computing the pointwise accuracy against a ground truth mask. Visualizations of the original MRI, segmented masks, and frequency distributions of pixel intensities should be provided. Additionally, the fitted GMM distributions and regions of misclassification should be analyzed.

## 3. Principal Component Analysis (PCA) (q3)

### 3.1 Explained Variance and Reconstruction 

PCA should be implemented from scratch and applied to a sample of MNIST images. The images must be projected onto different numbers of principal components (500, 300, 150, and 30). Plots of the explained variance ratio and scatter plots of the first two components must be created. Reconstructed images at each dimensionality should be visualized and compared to the original.

### 3.2 Classification Performance after PCA 

An MLP classifier should be trained on the MNIST dataset in its original form and after applying PCA with different numbers of components. Classification metrics such as accuracy, precision, and recall should be recorded. The experiment aims to analyze how dimensionality reduction affects model performance at various compression levels.

## 4. Autoencoder for Anomaly Detection (q4)

Implement an Autoencoder using PyTorch to reconstruct images of a selected digit class from MNIST. Use reconstruction errors to detect anomalies. Evaluate the model using precision, recall, and F1-score. Plot histograms of reconstruction errors for both normal and anomalous samples. Repeat the experiment with three different bottleneck layer sizes, and plot ROC curves to compare AUC-ROC scores for each configuration.

### 5. Variational Autoencoder (VAE) (q5)

Implement a Variational Autoencoder (VAE) using PyTorch for the same anomaly detection task on MNIST. The VAE must generate a latent representation by learning a probabilistic distribution (mean and variance). Use the reconstruction loss and KL-divergence to train the model. Compare the anomaly detection performance of the VAE with that of the standard Autoencoder by plotting ROC curves and comparing AUC-ROC scores. Discuss the differences in performance and latent space behavior.


