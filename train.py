from __future__ import division,print_function,absolute_import
from resnet3d import Resnet3DBuilder

from keras.models import Sequential
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

#GPU Calling
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Parameters Definition
num_epoch=5
learning_rate=0.001

#Optimizer Definiton
sgd=optimizers.SGD(lr=learning_rate)

#Reading Training data
voxel_train=[]
seg_train=[]

for i in tqdm(range(584),desc='测试数据读取中：'):
    try:
        picture=np.load('./train_val/candidate{}.npz'.format(i))
    except FileNotFoundError:
        continue
    try:
        voxel_train=np.append(voxel_train,np.expand_dims(picture['voxel'],axis=0),axis=0)
        seg_train=np.append(seg_train,np.expand_dims(picture['seg'],axis=0),axis=0)
    except ValueError:
        voxel_train=np.expand_dims(picture['voxel'],axis=0)
        seg_train=np.expand_dims(picture['seg'],axis=0)

seg_train=np.array(seg_train,dtype=int)
data_train=voxel_train*seg_train         #Filtering

data_train=np.array(data_train,dtype=np.float32)
data_train/=255
data_train=data_train.reshape(465,100,100,100,1)

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

#Reading Label Data
data_csv=pd.read_csv('./train_val.csv',usecols=[1])
label_train=np.array(data_csv.values.tolist())
label_train=to_categorical(label_train,num_classes=2)

#Split Training data and Test data
X_train,X_test,Y_train,Y_test=train_test_split(data_train,label_train,test_size=0.3)

#Training
model = Resnet3DBuilder.build_resnet_101((100,100,100, 1), 2)
model.compile(loss="categorical_crossentropy", optimizer="sgd",metrics=['accuracy'])
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=num_epoch,batch_size=45)
loss,accuracy=model.evaluate(X_test,Y_test,verbose=0)

#Prediction
prediction=model.predict(data_test)

#Writing Data into .csv file
prediction_data=pd.DataFrame(prediction)
prediction_data.to_csv('./submission.csv')

#Save the model
model.save('./my_model.h5')
json_string=model.to_json()
open('./model_architecture_1.json','w').write(json_string)
yaml_string=model.to_yaml()
open('./model_architecture_2.yaml','w').write(yaml_string)
model.save_weights('./my_model_weights.h5')
