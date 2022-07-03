import os 
import numpy as np
from tensorflow.keras.layers import AveragePooling2D
def load_feature_data():
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    with open("UCI HAR Dataset/train/X_train.txt","r") as f:
        while True:
            line=f.readline()
            if line=="":
                break
            line=line.split()
            x_train.append(line)
    with open("UCI HAR Dataset/train/y_train.txt","r") as f:
        while True:
            line=f.readline()
            if line=="":
                break
            y_train.append(line)

    with open("UCI HAR Dataset/test/X_test.txt","r") as f:
        while True:
            line=f.readline()
            if line=="":
                break
            line=line.split()
            x_test.append(line)
    with open("UCI HAR Dataset/test/y_test.txt","r") as f:
        while True:
            line=f.readline()
            if line=="":
                break
            y_test.append(line)

    x_train=np.array(x_train,dtype="float32")
    y_train=np.array(y_train,dtype="float32")
    x_test=np.array(x_test,dtype="float32")
    y_test=np.array(y_test,dtype="float32")
    return x_train,y_train,x_test,y_test
def load_dataset(mode="train"):
    body_acc_x_list=[]
    body_acc_y_list=[]
    body_acc_z_list=[]
    total_acc_x_list=[]
    total_acc_y_list=[]
    total_acc_z_list=[]
    total_gyro_x_list=[]
    total_gyro_y_list=[]
    total_gyro_z_list=[]
    with open(r"UCI HAR Dataset/%s/Inertial Signals/body_acc_x_%s.txt"%(mode,mode)) as f:
        while True:
            line=f.readline()
            if line=="":
                break
            body_acc_x=line.split()
            body_acc_x_list.append(body_acc_x)
    with open(r"UCI HAR Dataset/%s/Inertial Signals/body_acc_y_%s.txt"%(mode,mode)) as f:
        while True:
            line=f.readline()
            if line=="":
                break
            body_acc_y=line.split()
            body_acc_y_list.append(body_acc_y)
    with open(r"UCI HAR Dataset/%s/Inertial Signals/body_acc_z_%s.txt"%(mode,mode)) as f:
        while True:
            line=f.readline()
            if line=="":
                break
            body_acc_z=line.split()
            body_acc_z_list.append(body_acc_z)
    with open(r"UCI HAR Dataset/%s/Inertial Signals/total_acc_x_%s.txt"%(mode,mode)) as f:
        while True:
            line=f.readline()
            if line=="":
                break
            total_acc_x=line.split()
            total_acc_x_list.append(total_acc_x)
    with open(r"UCI HAR Dataset/%s/Inertial Signals/total_acc_y_%s.txt"%(mode,mode)) as f:
        while True:
            line=f.readline()
            if line=="":
                break
            total_acc_y=line.split()
            total_acc_y_list.append(total_acc_y)
    with open(r"UCI HAR Dataset/%s/Inertial Signals/total_acc_z_%s.txt"%(mode,mode)) as f:
        while True:
            line=f.readline()
            if line=="":
                break
            total_acc_z=line.split()
            total_acc_z_list.append(total_acc_z)
            
    with open(r"UCI HAR Dataset/%s/Inertial Signals/body_gyro_x_%s.txt"%(mode,mode)) as f:
        while True:
            line=f.readline()
            if line=="":
                break
            total_gyro_x=line.split()
            total_gyro_x_list.append(total_gyro_x)
            
    with open(r"UCI HAR Dataset/%s/Inertial Signals/body_gyro_y_%s.txt"%(mode,mode)) as f:
        while True:
            line=f.readline()
            if line=="":
                break
            total_gyro_y=line.split()
            total_gyro_y_list.append(total_gyro_y)
            
    with open(r"UCI HAR Dataset/%s/Inertial Signals/body_gyro_z_%s.txt"%(mode,mode)) as f:
        while True:
            line=f.readline()
            if line=="":
                break
            total_gyro_z=line.split()
            total_gyro_z_list.append(total_gyro_z)

    with open(r"UCI HAR Dataset/%s/y_%s.txt"%(mode,mode)) as f:
        y_train=f.readlines()
        y_train=np.array(y_train).astype("int32")

    x_train=np.stack([
        body_acc_x_list,
        body_acc_y_list,
        body_acc_z_list,
        total_acc_x_list,
        total_acc_y_list,
        total_acc_z_list,
        total_gyro_x_list,
        total_gyro_y_list,
        total_gyro_z_list,
    ],axis=-1).astype("float32")
    
    body_acc_length= (x_train[...,0]**2+x_train[...,1]**2+x_train[...,2]**2)**0.5
    total_acc_length=(x_train[...,3]**2+x_train[...,4]**2+x_train[...,5]**2)**0.5
    gyro_acc_length= (x_train[...,6]**2+x_train[...,7]**2+x_train[...,8]**2)**0.5
    
    x_train=np.concatenate([
        x_train,
        body_acc_length[...,None],
        total_acc_length[...,None],
        gyro_acc_length[...,None],
    ],axis=-1)
    return x_train,y_train-1
def load_sequences_data():
    x_train,y_train=load_dataset(mode='train')
    x_test,y_test=load_dataset(mode='test')
    return x_train,y_train,x_test,y_test
def load(mode:['feature','sequence']):
    assert(mode in ['feature','sequence'])
    if mode=='feature':
        return load_feature_data()
    if mode=='sequence':
        x_train,y_train,x_test,y_test= load_sequences_data()
        x_train=x_train[...,None]
        x_test=x_test[...,None]
        x_test=AveragePooling2D([2,1],[2,1])(x_test).numpy()
        x_train=AveragePooling2D([2,1],[2,1])(x_train).numpy()
        return x_train,y_train,x_test,y_test
