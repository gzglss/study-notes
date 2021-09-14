import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *

def scale_attention(x,d):
    x=x/d
    return x
    
class SigleAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,scale=True,**kwargs):
        super(SigleAttention,self).__init__()
        self.d=d_model
        self.dense=Dense(d_model)
        self.scale=scale
        
    def build(self,input_shape):
        super(SigleAttention,self).build(input_shape)
        
    def call(self,x):
        q=self.dense(x)
        k=self.dense(x)
        v=self.dense(x)
        
        qkt=tf.matmul(q,k,transpose_b=True)
        if self.scale:
            qkt=scale_attention(qkt,np.sqrt(x.shape[-1]))
        a=tf.nn.softmax(qkt)
        assert len(a.shape)==len(x.shape)
        ax=tf.matmul(a,x)
        return ax
    
    def get_config(self):
        config=super(SigleAttention,self).get_config()
        config.update({'d_model':self.d_model,
                      'scale':self.scale})
        return config
        
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,head_num,scale=True,**kwargs):
        super(MultiHeadAttention,self).__init__()
        self.h=head_num
        assert d_model%head_num==0
        self.d=d_model//head_num
        self.h=head_num
        self.head_num=head_num
        self.scale=scale
        
    def build(self,input_shape):
        super(MultiHeadAttention,self).build(input_shape)
        
    def call(self,x):
        ax=[]
        for i in range(self.h):
            x_h=tf.slice(x,[0,0,i*self.d],[x.shape[0],x.shape[1],self.d])
            attn=SigleAttention(self.d)
            attn.build(input_shape=(x.shape[1],self.d))
            a_h=attn(x_h)
            ax.append(a_h)
        ax_c=tf.concat(ax,axis=-1)
        return ax_c
    
    def get_config(self):
        config=super(MultiHeadAttention,self).get_config()
        config.update({'d_model':self.d_model,
                      'head_num':self.head_num,
                      'scale':self.scale})
        return config

#eg      
x=np.random.random((2,3,8))
a=MultiHeadAttention(8,2)
a.build((3,8))
a(x)