##  CABNet

This repository contains the Keras implementation using Tensorflow as backend of our paper "CABNet: Category Attention Block for Imbalanced Diabetic Retinopathy Grading"

## Requirements

python 3.6

numpy 1.16.4

keras 2.3.1

tensorflow 1.13.1

pillow 7.0.0

opencv-python 4.1.0

## Trained models

The trained model ： Baidu Yun : https://pan.baidu.com/s/1G2tKYNvqP7jmCl-svC1cfA  password: kjga
Google Drive：https://drive.google.com/drive/folders/1GNn3tj7WTLPjdOJUZqItui58DDQtYrsS?usp=sharing



## Usage

1. Clone the repository, and download the trained model, put them into 'weights' folder, you can run test.py to test the model directly. 
   The details of the trained model are listed in details of trained models.txt file.

2. If you want to train the model, download the dataset and put them into`data` folder.

3.  And then run the code：python train.py
    Note that the parameters and paths should be set beforehand

4. Once the training is complete, you can run the test.py to test your model.
   Run the code : python test.py.

## LICENSE
 Code can only be used for ACADEMIC PURPOSES. NO COMERCIAL USE is allowed.
 Copyright © College of Computer Science, Nankai University. All rights reserved.

## Citation
Along He, Tao Li, Ning Li, Kai Wang, and Huazhu Fu. CABNet: Category Attention Block for Imbalanced Diabetic Retinopathy Grading. IEEE Transactions on Medical Imaging, 2020.
