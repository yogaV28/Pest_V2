# Optimized YOLOv8 Architecture for Insect Detection at Agricultural Farms using Intelligent Edge Vision Systems (IEVSs)

## Overview

This project aims to develop and deploy an Intelligent Edge Vision System (IEVS) for real-time insect detection and classification in agricultural farms. Leveraging the YOLOv8 model optimized with TensorRT on the Jetson Nano platform, this system provides an efficient and scalable solution for precision agriculture.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Datasets](#datasets)
7. [Model Training](#model-training)
8. [Optimization](#optimization)
9. [Inference](#inference)
10. [Results](#results)
11. [Contributors](#contributors)
12. [License](#license)

## Introduction
Farmers face significant challenges due to insect infestations which can lead to substantial crop losses and economic burdens. This project introduces a real-time insect detection system using optimized YOLOv8 on edge devices to provide early detection and minimize crop damage, thereby promoting sustainable agricultural practices.

## Features
- Real-time insect detection and classification
- Optimized YOLOv8 model for edge devices using TensorRT
- High inference speed with low memory requirements
- Precision, recall, and F1-scores exceeding 95%
- Scalable and deployable on Jetson Nano devices

## System Requirements
- NVIDIA Jetson Nano
- TensorRT 7.0 or higher
- Python 3.6 or higher
- OpenCV
- PyTorch

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yogaV28/Pest_dectection
   cd Pest_dectection
   ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```
4. Set up Jetson Nano environment:
   Follow the instructions provided by NVIDIA to set up the Jetson Nano and install TensorRT.

## Usage:
1. **Data Preparation:** Collect images of insects and non-insects. Organize them into training and validation sets.
2. **Model Training:** Train the YOLOv8 model using the prepared dataset.
3. **Model Optimization:** Optimize the trained model using TensorRT for deployment on the Jetson Nano.
4. **Deployment:** Deploy the optimized model on the Jetson Nano and start the real-time insect detection.

##Datasets
The dataset should include a diverse set of insect images captured in various agricultural settings. Data augmentation techniques are recommended to enhance the model's performance on real-time data.
### Forty Insect classes in the NBAIR insect dataset (Cropped and Preprocessed)
![cs-102403-Figure_2](https://github.com/yogaV28/Pest_dectection/assets/121656366/290bf37f-9c47-4c43-a54c-fd0b3267d0f1)


## Model Training
1. Prepare the dataset and configure the training parameters.
2. Train the YOLOv8 model using the provided training scripts.
3. Save the trained model for further optimization.

## Optimization
1. Use TensorRT to optimize the trained YOLOv8 model.
2. Quantize the model to reduce memory usage and improve inference speed.
![image](https://github.com/yogaV28/Pest_dectection/assets/121656366/27076fe5-564b-43db-8adc-6d2134517821)

   
## Inference
1. Load the optimized model on the Jetson Nano.
2. Integrate the camera feed for real-time insect detection.
3. Run the inference script to start detecting insects.

## Results
The optimized YOLOv8 model achieved an inference speed of 45 fps on the Jetson Nano.
High accuracy with precision, recall, and F1-scores exceeding 95%.

### Training (Top row) and Validation loss (Bottom row) for the optimized YOLOv8 model with the augmented Insects dataset
![cs-102403-Figure_6](https://github.com/yogaV28/Pest_dectection/assets/121656366/af4574ac-d2a3-44dd-b3d2-ce162df023fa)
### Precision(Left), F1-Score(Middle) and Recall(Right) metrics of the model
![cs-102403-Figure_9](https://github.com/yogaV28/Pest_dectection/assets/121656366/8d627d3c-d40c-4402-af06-ab89fbdfcd1c)
### Precision-Recall Curve(Left) and Confusion Matrix(Right) for the proposed model
![cs-102403-Figure_10](https://github.com/yogaV28/Pest_dectection/assets/121656366/4cb21f01-2283-468f-bfbc-cc4f8e817410)
### Results of test cases with Predicted labels(left) and True labels(right)
![image](https://github.com/yogaV28/Pest_dectection/assets/121656366/a26d93ab-ab47-4a05-a688-56562e1a6a11)


## Contributors
Balaji Ganesh Rajagopal - SRM Institute of Science and Technology

Jagadeesh Kannan R - SRM Institute of Science and Technology

Yoga Vignesh V - SRM Institute of Science and Technology

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
