import tensorflow as tf
import pandas as pd
import numpy as np

class MyAttention(tf.keras.layers.Layer):
    def __init__(self,W_regularizers=None,\
                 b_regularizers=None,bias=True,\
                 **kwargs):
        super(MyAttention,self).__init__(**kwargs)
        self.W_regularizers=tf.keras.regularizers.get(W_regularizers)
        self.b_regularizers=tf.keras.regularizers.get(b_regularizers)
        self.init=tf.keras.initializers.get('glorot_uniform')
        self.bias=bias
        self.d_model=0
        self.mask=True
        
    def build(self,input_shape):
        self.timestep=input_shape[1]
        assert len(input_shape)==3#[batch_size,seq_len,d_model]
        self.w=self.add_weight(shape=(input_shape[-1],),\
                              initializer=self.init,\
                              name='{}_w'.format(self.name),\
                              regularizer=self.W_regularizers,)
        self.d_model=input_shape[-1]
        if self.bias:
            self.b=self.add_weight(shape=(input_shape[1],),\
                                  initializer='zeros',\
                                  name='{}_b'.format(self.name),\
                                  regularizer=self.b_regularizers,)
        else:
            self.b=None
            
        self.built=True
        
    def compute_mask(self,inputs,input_mask=None):
        return None
    
    def call(self,x,mask=None):#[batch_size,seq_len,d_model]
        d_model=self.d_model
        timestep=self.timestep
        x_=tf.reshape(x,(-1,d_model))#x.shape=[batch_size*seq_len,d_model]
        self.w=tf.reshape(self.w,(-1,1))#w.shape:[d_model,1]
        ei=tf.matmul(x_,self.w)
        ei=tf.reshape(ei,(x.shape[0],x.shape[1],1))#[batch_size*seq_len,1]
        self.b=tf.reshape(self.b,(-1,1))
        if self.bias:
            ei=ei+self.b
        ei=tf.tanh(ei)
        a=tf.exp(ei)
        a=tf.reshape(a,(x.shape[0],x.shape[1],1))#[batch_size,seq_len,1]
        if mask:#[batch_size,seq_len,1]需要mask的地方为0，其他为1
            a=tf.multiply(a,mask)
        a/=tf.reduce_sum(a,axis=1,keepdims=True)#注意力得分
        x_attn=tf.multiply(x,a)#[batch_size,seq_len,d_model]
        return x_attn
    
    def compute_output_shape(self,input_shape):
        return input_shape[0],input_shape[1],self.d_model
    
    def get_config(self):#使模型便于保存
        config=super(MyAttention,self).get_config()
        config.update({'W_regularizers':self.W_regularizers,
                     'b_regularizers':self.b_regularizers,
                     'bias':bias})
        return config
        
a=MyAttention()
a.build(input_shape=b.shape)
c=a(b)
