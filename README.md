# Machine Learning Team 
![TensorFlow](https://github.com/user-attachments/assets/670f65bd-9fea-4912-9a38-08267c202c3a)

## Project Name : Belens 

## Overview
This repository contains the code for **Belens**, a health-focused application aimed at assisting users in Indonesia in monitoring sugar consumption from packaged beverages. The project utilizes TensorFlow and lightweight Convolutional Neural Networks (CNN) for accurate logo recognition on beverage packaging.

## Dataset
The dataset includes a diverse collection of images featuring various packaged beverages available in Indonesia. Preprocessing steps, such as resizing, normalization, and augmentation, are implemented to ensure optimal model training and generalization.
Our dataset comprises a total of **30 products**, with **at least 40 images per product**, annotated with bounding boxes for accurate logo detection. This well-curated dataset ensures robust training and evaluation of the Convolutional Neural Network (CNN) model.

## Architecture
The primary model for beverage logo recognition utilizes a custom Convolutional Neural Network (CNN) architecture. The input images are resized to **120 x 120 x 3** to accommodate a smaller, more efficient model while maintaining performance. The model architecture includes several convolutional layers followed by pooling layers, and a few dense layers for final classification.

## Performance

### Accuracy
![ACCURACY](https://github.com/user-attachments/assets/33d486f1-2cc9-40f0-8b3b-d8d61014cef4)

### Loss
![LOSS](https://github.com/user-attachments/assets/ac0b9895-e17d-4b40-9c1f-d6cd146630d5)

### Classification Report
![KLASIFIKASIREPORT](https://github.com/user-attachments/assets/b3685d6d-3e05-4201-8567-e8f2b9f16d74)

## Run The Model

Requirement  : tensorflow, matplotlib, numpy, ipywidgets, pandas, io, ipython and pillow.

1. Open the CAPSTONE-PROYEK(1).ipnyb and execute all cells sequentially from top to bottom. THe process  indeed :
   - Importing all library needed such as tensorflow, matplotlib, numpy, ipywidgets, pandas, io, ipython and pillow.
   - import dataset from kaggle and unzipped it then load the directory no need to set the dataframe because the dataset already with structured folder.
   - Splitting the data into training and validation sets.
   - Construct an ImageDataGenerator for data augmentation.
   - Builded your own CNN architecture.
   - Compiling and training the model.
   - Displaying the model’s performance results.
   - Saving the trained model as “best_model.keras”.
2. The model saved as best_model.keras
   
