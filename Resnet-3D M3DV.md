# Resnet-3D M3DV

> Machine Learning Project M3DV
>
> by : Wu Ningyuan
>
> Contact: williamxeon@sjtu.edu.cn



## Brief Description

Kaggle Competition Name: SJTU M3DV: Medical 3D Voxel Classification

Website: https://www.kaggle.com/c/sjtu-m3dv-medical-3d-voxel-classification/

Dataset and brief introduction can be accessed from the website as above.

The tested accuracy is 64% at most (accuracy may vary from time to time).



## Version Information

keras==2.3.1

Tensorflow==1.13.1

Python==3.7.0



## Relevant Dataset

train_val(465 .npz files)

test(117 .npz files)



## Installation

Anaconda 3

Tensorflow

Keras



## Usage

1. open the file 'train.py' ;
2. Read the dataset with numpy ;
3. Change the type of Resnet and click 'run'
4. Wait and the final result will be updated



## Special Thanks

This implementation will give special thanks to JihongJv for his Resnet 3d Implementation Code and SJTUSEIEECBX for his data-processing code.

website: https://github.com/JihongJu/keras-resnet3d 

website: https://github.com/SJTUSEIEECBX/m3dv/blob/master/DataProcessing.py