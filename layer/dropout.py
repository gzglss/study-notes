class Dropout(Layer):
    def __init__(self,p=0.2):
        self.p=p
        self._mask=None
        self.input_shape=None
        self.trainable=True
        self.n_units=None
        self.pass_throught=True

    def forward_pass(self,x,training=True):
        c=1-self.p
        if training:
            self._mask=np.random.uniform(size=x.shape)>self.p
            x/=c
            #dropout的rescale，保证网络在预测的时候的输入与训练时候的输入的一致性，使得训练集的输出可以对测试集起到
            #预测的作用，在训练的时候除以c是为了避免在预测的时候乘以c
            c=self._mask
        return x*c

    def backward_pass(self,accm_grad):
        #不计算失活的神经元的梯度
        return accm_grad*self._mask
        
    def output_shape(self,):
        return self.input_shape
        
        
        
        
def dropout(x,level):
    if level<0 or level>1:
        raise Exception("dropout level must in [0,1]")
    retain_prob=1-level
    sample=np.random.binomial(n=1,p=retain_prob,size=x.shape)
    #二项分布--伯努利分布
    x*=sample
    x/=retain_prob#rescale
    return x