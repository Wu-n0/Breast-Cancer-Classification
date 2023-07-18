# Breast Cancer Classification Project

## Project Description

This Python project aims to build a breast cancer classifier using the Invasive Ductal Carcinoma (IDC) dataset. The objective is to accurately classify histology images as either benign or malignant. The project utilizes Convolutional Neural Networks (CNNs) implemented with Keras to achieve this classification task. The performance of the model is analyzed using a confusion matrix.

## Prerequisites

Ensure you have the following Python packages installed before running the project:

- numpy
- opencv-python
- pillow
- tensorflow
- keras
- imutils
- scikit-learn
- matplotlib

## Dataset

The dataset used in this project focuses on Invasive Ductal Carcinoma (IDC), which is the most prevalent subtype of breast cancer. Pathologists commonly determine the aggressiveness grade of a whole mount breast cancer sample by identifying regions containing IDC. Consequently, a crucial pre-processing step for automatic aggressiveness grading involves delineating these IDC regions within the whole mount slides.

The original dataset comprises 162 whole-mount slide images of Breast Cancer (BCa) specimens scanned at 40x magnification. From these images, a total of 277,524 patches, each measuring 50 x 50 pixels, were extracted. Among these patches, 198,738 are IDC negative, while 78,786 are IDC positive. Each patch's filename follows the format "u_xX_yY_classC.png," where "u" represents the patient ID (e.g., 10253_idx5), "X" denotes the x-coordinate of the cropped patch, "Y" indicates the y-coordinate of the cropped patch, and "C" signifies the class label (0 for non-IDC and 1 for IDC).

## Getting Started

1. Download the zip file from [https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images]
2. Create a new directory named "datasets" to place the dataset.
3. Inside the "datasets" directory, create a new folder named "original."
4. Download the IDC_regular dataset from Kaggle and unzip it into the "original" directory.
5. Run the "prepare_dataset.py" script to preprocess the dataset and split it into training, validation, and testing sets.
6. Finally, run the "train_classifier.py" script to train the BreastCancerClassifier model and evaluate its performance.

## Data Preprocessing and Dataset Splitting

To prepare the dataset for training, validation, and testing, use the "prepare_dataset.py" script in the "BreastCancerClassifier" directory. This script sets up the directory structure and paths for organizing the breast cancer histology image dataset. It designates the input dataset directory as "datasets/original" and creates new directories for training, validation, and testing data under "datasets/idc". The dataset is then split into 80% training, 10% validation, and 10% testing sets, which are saved in their respective directories for subsequent use in the breast cancer classifier.

## BreastCancerClassifier Model

The "breast_cancer_classifier.py" file contains the definition of the BreastCancerClassifier model, a Convolutional Neural Network (CNN) designed for breast cancer classification. The model uses 3x3 convolutional filters stacked together, followed by max-pooling and depthwise separable convolution operations. This approach optimizes memory usage while maintaining efficiency in the model's feature extraction process. The CancerNet class is constructed using the TensorFlow Sequential API, and the depthwise convolution is implemented through the SeparableConv2D layer. The "build()" method of the CancerNet class takes the image width, height, depth, and the number of classes to predict (in this case, 2 classes: 0 and 1) as input parameters. The method initializes the model, defines the model architecture with multiple depthwise convolutional layers, applies activation functions, and finally returns the constructed model for breast cancer classification.

## Training and Evaluation

The "train_classifier.py" script trains and evaluates the BreastCancerClassifier model for breast cancer classification. It imports necessary libraries and sets the initial values for epochs, learning rate, and batch size. The dataset is split into training, validation, and testing sets, and class weights are calculated to handle class imbalance. Data augmentation is applied to the training data for regularization. The BreastCancerClassifier model is initialized with the Adagrad optimizer and binary cross-entropy loss. The model is then trained using the fit() method with the training and validation data generators. After training, the model is evaluated on the testing data, and a classification report is displayed showing metrics like precision, recall, and F1-score for each class. The confusion matrix is computed, and accuracy, specificity, and sensitivity are calculated and displayed. Finally, the training loss and accuracy are plotted to visualize the model's performance during training.


