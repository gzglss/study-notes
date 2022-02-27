class Mha(tf.keras.layers.Layer):
    def __init__(self,d_model,head_nums,scale=True,dropout=0.2,**kwargs):
        super(Mha,self).__init__()
        self.h=head_nums
        self.scale=scale
        self.dropout_rate=dropout
        self.d_model=d_model
        assert d_model%head_nums==0
        self.d=d_model//head_nums
        self.wq=tf.keras.layers.Dense(d_model)
        self.wk=tf.keras.layers.Dense(d_model)
        self.wv=tf.keras.layers.Dense(d_model)
        self.dropout=tf.keras.layers.Dropout(dropout)
        
    def scale_dot_product_attention(self,q,k,v,mask,dim,scale):
        qkt=tf.matmul(q,k,transpose_b=True)
        if scale:
            qkt=qkt/tf.math.sqrt(dim)
        if mask:
            #需要注意的是bert的输入mask中0表示填充的文本
            mask=tf.keras.ons_like(mask)-mask
            qkt+=mask*(-1e9)
        return qkt

    def build(self,input_shape):
        super(Mha,self).build(input_shape)
        
    @tf.function    
    def call(self,mask=None,x,dropout=None):
        x_h=tf.reshape(x,(x.shape[0],x.shape[1],self.h,self.d))
        x_h=tf.transpose(x,[0,2,1,3])#[batch_size,head_nums,seq_len,d]
        q=self.wq(x_h)
        k=self.wk(x_h)
        v=self.wv(x_h)
        scale_attn=scale_dot_product_attention(q,k,v,mask,self.d,self.scale)
        scale_attn=tf.nn.softmax(scale_attn)
        if dropout:
            scale_attn=self.dropout(scale_attn)
        a=tf.matmul(scale_attn,v)#[batch_size,head_nums,seq_len,d]
        a=tf.reshape(a,(a.shape[0],a.shape[2],self.h*self.d))#[batch_size,seq_len,d_model]
        return a
        
    def get_config(self):
        config=super(Mha,self).get_config()
        config.update({"d_model":self.d_model,
                       "head_nums":self.h,
                       "scale":self.scale,
                       "dropout":self.dropout_rate})
        return config