# Brain Tumor Classification Using Machine Learning

## Overview
This project focuses on developing and evaluating machine learning models for classifying brain MRI images to detect brain tumors. The goal is to achieve high accuracy and reliability in tumor detection using various deep learning architectures. Additionally, an interactive web-based interface is implemented to assist medical professionals in uploading MRI images and receiving classification results.

## Features
- **Multiple Machine Learning Architectures**: Custom CNN, pre-trained CNN (VGG19, EfficientNetB0), and Vision Transformer.
- **Dataset Selection & Preprocessing**: MRI images sourced from Kaggle, resized, and normalized.
- **Model Evaluation**: Comparative analysis based on validation accuracy and test accuracy.
- **Interactive Web Interface**: Implemented using Gradio for real-time image classification and visualization.
- **Hyperparameter Optimization**: Conducted using Weights and Biases (wandb).
- **Grad-CAM Heatmap**: Provides visual explanation for model predictions.

## Project Objectives
1. **Develop an Accurate Tumor Classification Model**: Classify MRI images into four categories: Glioma Tumor, Pituitary Tumor, Meningioma Tumor, and No Tumor.
2. **Compare Machine Learning Models**: Evaluate the performance of different architectures.
3. **Deploy the Best Model**: Integrate the best-performing model into a web application.
4. **Provide Interpretability**: Use Grad-CAM for model explainability.

## Dataset
- **Source**: Kaggle - Brain MRI Tumor Classification Dataset
- **Classes**: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, No Tumor
- **Training Images**: 2870
- **Test Images**: 394
- **Preprocessing Steps**:
  - Image resizing (224x224 pixels)
  - Conversion to NumPy arrays
  - Normalization

## Model Architectures
### 1. **Custom CNN**
- Developed a convolutional neural network tailored for MRI classification.
- Optimized using hyperparameter tuning.

### 2. **Pre-trained CNN Models**
- **VGG19**: Transfer learning applied with fine-tuning.
- **EfficientNetB0**: Selected for better computational efficiency and accuracy.

### 3. **Vision Transformer (ViT)**
- Uses self-attention mechanisms for classification.
- Divides images into patches and processes them using Transformer architecture.

## Hyperparameter Tuning
- **Tool Used**: Weights and Biases (wandb)
- **Optimization Method**: Random Search
- **Hyperparameters Tuned**:
  - Learning Rate
  - Batch Size
  - Dropout
  - Number of Epochs
  - Optimizer (Adam, RMSprop, SGD)

## Evaluation Metrics
- **Validation Accuracy**
- **Test Accuracy**
- **Computational Efficiency**

### Model Performance Summary:
| Model               | Validation Accuracy | Test Accuracy |
|---------------------|--------------------|--------------|
| Custom CNN         | 88.50%              | 70.30%       |
| VGG19 Pre-trained  | 89.37%              | 64.72%       |
| EfficientNetB0     | 97.56%              | 79.44%       |
| Vision Transformer | 32.23%              | 23.22%       |

## Deployment
- **Web App**: Built using **Gradio**
- **Features**:
  - Upload MRI images
  - Get tumor classification results
  - View Grad-CAM heatmaps for explainability

## Installation & Usage
### Prerequisites
- Python 3.8+
- TensorFlow
- PyTorch
- NumPy, OpenCV
- Gradio
- Weights and Biases (wandb)


## Future Improvements
- Explore multi-modal data fusion techniques.
- Optimize Vision Transformer implementation.
- Improve web app user experience and performance.
- Deploy the model using AWS or Azure.

## Contributors
- **Karan Balasundaram**  

## License
This project is released under the MIT License.


