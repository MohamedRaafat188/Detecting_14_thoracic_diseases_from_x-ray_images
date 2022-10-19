# Detecting_14_thoracic_diseases_from_x-ray_images using CNN


## Table of Contents
1. [Description](#Description)
2. [Objective](#Objective)
3. [Dataset](#Dataset)
4. [Preprocessing](#Preprocessing)
5. [ML Pipeline](#ML Pipeline)
6. [Installation](#Installation)


### Description
Diagnosis of diseases from medical images faces many challenges and we notice how AI influences every aspect in our life so I tried to use deep learning (CNN) to classify x-ray images.
In October 2017, the National Institute of Healthcare open sourced 112,000+ images of chest x-rays. Now known as ChestXray14, this dataset was opened in order to allow clinicians to make better diagnostic decisions for patients with various lung diseases.


### Objective 
The aim of this project for me is to learn how to deal with the multi-label classification tasks so the results may not be very good but i will try to improve it later. In this project I learned how to train and validate a model in multi-label classification tasks using pytorch lightning library. I learned the following:
* build a custom dataset class
* which loss function should be used (Binary Cross Entropy)
* how to calculate evaluation metrics such as recall, precision, accuracy and specificity
* how to use pytorch lightning library in traing, validation and testing the model
* how to use torchmetrics library to calculate different metrics
* how to use wandb library to track my model performance
* how to write good and fully descriptive readme file

The goal from the previous points is to build a CNN that can classify chest x-ray images.


### Dataset
The ChestXray14 [dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345) consists of both images and structured data. I will use only the images. Later i will try to benefit from the structured data.

The image dataset consists of 112,000 images, which consist of 30,000 patients. Some patients have multiple scans. All images are originally 1024*1024 pixels.
I will use only 51000+ images of the data because the dataset has about 61000 images have no disease so we have to reduce the data imbalance and that will make the training process much faster.


### Preprocessing
The images are of size 1024*1024 and in .png format so I applied the following steps:
* convert the images to grayscale images
* resize the images to the size of 224*224 because my pc is not strong enough to deal with larger size but it's better to increase the size
* normalize the images
* save the images in .npy format using numpy library


### ML Pipeline
* create a custom dataset class
* apply some data augmentation techniques (random rotation, random translation, random scaling and random cropping)
* use dataloaders
* visualize some of the data after being augmented
* create model class using pytorch lightning to handle training, validation and testing processes
    * efficientnet_b4 pretrained model as a feature extractor and a custom fully connected part 
    * Adam optimizer
    * BCEWithLogitsLoss loss function 
* create some checkpoints to save the best weights based on some criteria
* use wandb library to track the model performance


### Installation
You must have the latest versions(October 2022) of these libraries:
* numpy
* pandas
* matplotlib
* cv2
* sklearn
* tqdm
* pytorch
* pytorch_lightning
* torchmetrics
* wandb

And of course you should have python 3.8.5 and jupyter
