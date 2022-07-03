#网络结构
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
class SoftThreshold(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(SoftThreshold,self).__init__(**kwargs)
    def build(self,input_shape):
        super(SoftThreshold,self).build(input_shape)
    def call(self,inputs):
        tensor=inputs[0]
        threshold=inputs[0]*inputs[1]
        threshold=tf.abs(threshold)
        tensor=tf.where(tensor<-threshold,tensor+threshold,0)+tf.where(tensor>threshold,tensor-threshold,0)
        return tensor
class SIModule(object):
    def __init__(self,filters,kernel_size,activation,dilation_rates=[1,1,1,1]):
        self.filters=filters
        self.kernel_size=kernel_size
        self.dilation_rates=dilation_rates
        self.activation=activation
        self.sk_rate=8
    def branch(self,tensor,dilation_rate):
        x=Conv2D(self.filters,self.kernel_size,1,padding="same",kernel_regularizer=l2(0.00),
                 activation=self.activation,dilation_rate=dilation_rate)(tensor)
        x=BatchNormalization()(x)
        x=Activation(None)(x)
        return x
    def attention(self,tensor):
        add=tf.keras.layers.add(tensor)
        shape=add.shape.as_list()
        squeeze=GlobalAveragePooling2D()(add)
        squeeze=Dense(shape[-1]//self.sk_rate,activation=tf.nn.relu)(squeeze)
        extracts=tf.stack([Dense(shape[-1],activation=None)(squeeze) for i in range(len(tensor))],axis=-1)
        extracts=tf.nn.softmax(extracts,axis=-1)
        extracts=tf.split(extracts,len(tensor),axis=-1)
        output=[]
        for x,t in zip(tensor,extracts):
            t=tf.expand_dims(t[...,0],axis=1)
            t=tf.expand_dims(t,axis=1)*0.01
            output.append(SoftThreshold()([x,t]))
        return output
    def pool(self,tensor):
        tensor=Conv2D(self.filters,self.kernel_size,1,padding="same",kernel_regularizer=l2(0.00))(tensor)
        shape=tensor.shape.as_list()#[batch,lenth,channels]
        scaler=AveragePooling2D((shape[1],shape[2]))(tensor)#[batch,channels]
        scaler=UpSampling2D((shape[1],shape[2]))(scaler)
        return scaler
    def __call__(self,tensor):
        x=Conv2D(self.filters,1,1,padding="same",kernel_regularizer=l2(0.00))(tensor)
        x=BatchNormalization()(x)
        x=Activation(None)(x)
        output=[x]
        output.append(self.pool(tensor))
        for rate in self.dilation_rates:
            output.append(self.branch(tensor,rate))
        output=self.attention(output)
        output=tf.keras.layers.add(output)
        output=Conv2D(self.filters,3,1,padding="same",kernel_regularizer=l2(0.00))(output)
        output=BatchNormalization()(output)
        output=Activation(self.activation)(output)
        return output
    
def _SINet(inputs,num_classes,dilation_rates=[1,1,1,1]):
    x=Conv2D(32,3,1,padding="same",kernel_regularizer=l2(0.00))(inputs)
    x=BatchNormalization()(x)
    x=Activation(tf.nn.relu)(x)
    
    x=Conv2D(64,3,1,padding="same",kernel_regularizer=l2(0.00))(x)
    x=BatchNormalization()(x)
    x=Activation(tf.nn.relu)(x)
    x=Dropout(0.3)(x)
    
    x=Conv2D(196,3,1,padding="same",kernel_regularizer=l2(0.00))(x)
    x=BatchNormalization()(x)
    x=Activation(tf.nn.relu)(x)
    
    x=MaxPooling2D(2,padding="same")(x)
    x=Dropout(0.5)(x)
    
    x=Conv2D(256,3,1,padding="same",kernel_regularizer=l2(0.00))(x)
    x=BatchNormalization()(x)
    x=Activation(tf.nn.relu)(x)
    x=Dropout(0.5)(x)
    
    x=SIModule(128,3,activation=tf.nn.relu,dilation_rates=dilation_rates)(x)
    x=Dropout(0.5)(x)
    
    x=GlobalAveragePooling2D()(x)
    x=Dropout(0.5)(x)

    x=Dense(num_classes,kernel_regularizer=l2(0))(x)
    x=Softmax()(x)
    return x

def _FCN(inputs,num_classes):
    x=Dense(512)(inputs)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    x=Dense(256)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    x=Dense(128)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    x=Dense(num_classes)(x)
    x=BatchNormalization()(x)
    x=Softmax()(x)
    return x
def SINet(input_shape=[64,12,1],num_classes=6,num_branches=5):
    assert(num_branches>4)
    inputs=Input(shape=input_shape)
    outputs=_SINet(inputs,num_classes=num_classes,dilation_rates=[1]*(num_branches-2))
    cnn_model=Model(inputs=inputs,outputs=outputs)
    return cnn_model
def FCN(input_shape=[561],num_classes=6):
    inputs=Input(shape=input_shape)
    outputs=_FCN(inputs,num_classes=num_classes)
    dnn_model=Model(inputs=inputs,outputs=outputs)
    return dnn_model
                  
