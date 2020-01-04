from __future__ import division,print_function,absolute_import
from resnet3d import Resnet3DBuilder

from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Flatten,Conv3D,MaxPool3D,BatchNormalization,Input
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau,TensorBoard

import pandas as pd 
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix,accuracy_score

import tensorflow as tf
import os

#Reading Test Data
voxel_test=[]
seg_test=[]

for j in tqdm(range(583),desc='测试数据读取中：'):
    try:
        picture_test=np.load('./test/candidate{}.npz'.format(j))
    except FileNotFoundError:
        continue
    try:
        voxel_test=np.append(voxel_test,np.expand_dims(picture_test['voxel'],axis=0),axis=0)
        seg_test=np.append(seg_test,np.expand_dims(picture_test['seg'],axis=0),axis=0)
    except ValueError:
        voxel_test=np.expand_dims(picture_test['voxel'],axis=0)
        seg_test=np.expand_dims(picture_test['seg'],axis=0)

seg_test=np.array(seg_test,dtype=int) 
data_test=voxel_test*seg_test    #Filtering

data_test=np.array(data_test,dtype=np.float32)
data_test/=255
data_test=data_test.reshape(117,100,100,100,1)

#Model Loading and Using
print("Using loaded model to predict...")
load_model=load_model("./权重文件/my_model.h5")
prediction=load_model.predict(data_test)
prediction_data=pd.DataFrame(prediction)
prediction_data.to_csv('./submission.csv')