import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *

def scale_attention(x,d):
    x=x/d
    return x
    
def mask_(x,mask,pad=0):
    #这里的mask一般以tokenizer输出的attention_mask作为输入
    #而attention_mask是未padding的部分取1，padding的部分取0
    #所以这里需要反向取mask
    #mask矩阵为1的地方就是需要mask的地方
    mask=1-mask
    assert len(mask.shape)==2
    mask=tf.multiply(mask,-1000000)
    mask=tf.expand_dims(mask,axis=1)
    mask=tf.expand_dims(mask,axis=-1)
    mask=tf.cast(mask,dtype=x.dtype)
    x_mask=tf.add(x,mask)
    return x_mask
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,head_num,scale=True,**kwargs):
        super(MultiHeadAttention,self).__init__()
        self.h=head_num
        self.d_model=d_model
        assert d_model%head_num==0
        self.d=d_model//head_num
        self.h=head_num
        self.head_num=head_num
        self.scale=scale
        self.dense=Dense(self.d)
        
    def build(self,input_shape):
        super(MultiHeadAttention,self).build(input_shape)
        
    def call(self,x,mask=None):
        x_h=tf.reshape(x,(x.shape[0],x.shape[1],self.h,self.d))
        x_h=tf.transpose(x_h,[0,2,1,3])#[batch_size,head_num,seq_len,d]
        q=self.dense(x_h)#[batch_size,head_num,seq_len,d]
        k=self.dense(x_h)
        v=self.dense(x_h)
        kt=tf.transpose(k,[0,1,3,2])
        qkt=tf.matmul(q,kt)#[batch_size,head_num,seq_len,seq_len]
        if self.scale:
            qkt=scale_attention(qkt,self.d)
        if mask is not None:
            qkt=mask_(qkt,mask)
        qkt=tf.nn.softmax(qkt)#[batch_size,head_num,seq_len,seq_len]
        a=tf.matmul(qkt,x_h)#[batch_size,head_num,seq_len,d]
        a=tf.reshape(a,(x.shape[0],x.shape[1],self.d_model))#[batch_size,seq_len,d_model]
        return a
    
    def get_config(self):
        config=super(MultiHeadAttention,self).get_config()
        config.update({'d_model':self.d_model,
                      'head_num':self.head_num,
                      'scale':self.scale})
        return config

#eg      
x=np.random.random((2,8,100))
b=np.array([[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0]])
a=MultiHeadAttention(100,4)
a.build((3,100))
a(x,mask=b).shape
