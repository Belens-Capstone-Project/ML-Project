# README: Machine Learning Team Project

## Project Description
**Project Name**: Belens 

## Overview
This repository contains the code for **Belens**, a health-focused application aimed at assisting users in Indonesia in monitoring sugar consumption from packaged beverages. The project utilizes TensorFlow and lightweight Convolutional Neural Networks (CNN) for accurate logo recognition on beverage packaging.

## Dataset
The dataset includes a diverse collection of images featuring various packaged beverages available in Indonesia. Preprocessing steps, such as resizing, normalization, and augmentation, are implemented to ensure optimal model training and generalization.
Our dataset comprises a total of **30 products**, with **at least 40 images per product**, annotated with bounding boxes for accurate logo detection. This well-curated dataset ensures robust training and evaluation of the Convolutional Neural Network (CNN) model.

## Architecture
The primary model for beverage logo recognition utilizes a custom Convolutional Neural Network (CNN) architecture. The input images are resized to **120 x 120 x 3** to accommodate a smaller, more efficient model while maintaining performance. The model architecture includes several convolutional layers followed by pooling layers, and a few dense layers for final classification.

## Performance
### Accuracy
### Loss
### Classification Report

## Run The Model

Requirement  : tensorflow, matplotlib, numpy, ipywidgets, pandas, io, ipython and pillow.
1. Clone the repository :
   git clone 
2. Installation :
3. Open the CAPSTONE-PROYEK(1).ipnyb and execute all cells sequentially from top to bottom. THe process  indeed :
   a. Importing all library needed such as tensorflow, matplotlib, numpy, ipywidgets, pandas, io, ipython and pillow.
   b. import dataset from kaggle and unzipped it then load the directory no need to set the dataframe because the dataset already with structured folder.
   c. Splitting the data into training and validation sets.
   d. Construct an ImageDataGenerator for data augmentation.
   e. Builded your own CNN architecture.
   f. Compiling and training the model.
   g. Displaying the model’s performance results.
   h. Saving the trained model as “best_model.keras”.
4. The model saved as best_model.keras
   
