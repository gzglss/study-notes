class Cnn:
    def __init__(self,num_filter,stride,kernel_size):
        super(self,Cnn).__init__()
        self.stride=stride
        self.ks=kernel_size
        self.filter_=np.random.uniform(num_filter,kernel_size[0],kernel_size[1])/(kernel_size[0]*kernel_size[1]))#除以一个数，减少初始方差
        
    def iter_regions(self,arr):
        h,w=arr.shape
        for i in range(h-self.ks[0]+1,self.stride):
            for j in range(w-self.ks[1]+1,self.stride):
                sub_arr=arr[i:i+self.ks[0]][j:j+self.ks[1]]
                yield sub_arr,i,j
    
    def forward(self,inputs):
        self.last_input=inputs
        h,w=inputs.shape
        output=np.zeros((h-self.ks[0]+1)/self.stride,(w-self.ks[1]+1)/self.stride)
        for sub_arr,i,j in self.iter_regions(inputs):
            output[i][j]=sum(sub_arr*self.filter_,axis=(1,2))
        return output
        
    def backward(self,loss_grad,lr):
        loss_grad_filter_=np.zeros(self.filter_.shape)
        for sub_arr,i,j in self.iter_regions(self.last_inputs):
            for f in range(self.num_filter):
                loss_grad_filter_[f]=sub_arr*loss_grad[i,j,f]#根据梯度确定输入的变化量
        self.filter_-=lr*loss_grad_filter#更新卷积核参数
        return loss_grad