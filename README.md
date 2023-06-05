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

The trained model ： Baidu Yun : https://pan.baidu.com/s/1G2tKYNvqP7jmCl-svC1cfA  password: kjga <br>
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
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=baaivision/Painter&type=Date)](https://star-history.com/#baaivision/Painter&Date)
## Citation
@article{he2020cabnet,<br/>
  title={CABNet: Category attention block for imbalanced diabetic retinopathy grading},<br/>
  author={He, Along and Li, Tao and Li, Ning and Wang, Kai and Fu, Huazhu},<br/>
  journal={IEEE Transactions on Medical Imaging},<br/>
  volume={40},<br/>
  number={1},<br/>
  pages={143--153},<br/>
  year={2020},<br/>
  publisher={IEEE}<br/>
}
