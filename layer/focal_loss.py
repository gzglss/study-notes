from keras import backend as K

#长尾分布，类别不平衡
def focal_loss(alpha=0.75, gamma=2.0):#（论文最后alpha=0.25, gamma=2.0）
    """ 参考 https://blog.csdn.net/u011583927/article/details/90716942 """
    def focal_loss_fixed(y_true, y_pred):
        # y_true 是个一阶向量, 下式按照加号分为左右两部分
        # 注意到 y_true的取值只能是 0或者1 (假设二分类问题)，可以视为“掩码”
        # 加号左边的 y_true*alpha 表示将 y_true中等于1的槽位置为标量 alpha
        # 加号右边的 (ones-y_true)*(1-alpha) 则是将等于0的槽位置为 1-alpha
        ones = K.ones_like(y_true)
        alpha_t = y_true*alpha + (ones-y_true)*(1-alpha)

        # 类似上面，y_true仍然视为 0/1 掩码
        # 第1部分 `y_true*y_pred` 表示 将 y_true中为1的槽位置为 y_pred对应槽位的值
        # 第2部分 `(ones-y_true)*(ones-y_pred)` 表示 将 y_true中为0的槽位置为 (1-y_pred)对应槽位的值
        # 第3部分 K.epsilon() 避免后面 log(0) 溢出
        p_t = y_true*y_pred + (ones-y_true)*(ones-y_pred) + K.epsilon()
	#p_t表示预测结果softmax后的概率，可以理解为预测结果的置信度，置信度愈大表明对该样本预测越简单，也需要降低该样本产生的loss
        # 就是公式的字面意思
        focal_loss = -alpha_t * K.pow((ones-p_t),gamma) * K.log(p_t)
        return focal_loss
    return focal_loss_fixed

model = ...
model.compile(..., loss=focal_loss(gamma=3, alpha=0.5))


#多分类问题的focal_loss
#核心思想：
#1.y_true,y_pred,alpha,gamma都是以向量的形式输入，当考虑batchsize=1时，向量的维度都是num_class
#2.y_true可以为[0,0,0,1,0,0,0],对应的y_pred可以为[0.05,0.05,0.1,0.6,0.1,0.05,0.05]
#3.将y_true与y_pred输入函数后，将其转化为二分类y_true_new=[0,1],y_pred_new=[0.4,0.6]
#4.此时alpha与gamma都取y_true中1对应的位置的值，此时就将一个multi_focal_loss转化为了focal_loss
#5.按照正常二分类的focal_loss计算loss
#单个样本
def multi_focal_loss(alpha,gamma):
    #注意多分类时，标签使用one-hot，y_true.shape=[1,num_class]
    #可以考虑将[0,0,0,1,0,0,0]处理为[0,1]的二分类形式
    def multi_focal_loss_fixed(y_true,y_pred):
        class_id=np.argmax(y_true)
        neg_ids=np.array([int(i) for i in range(len(y_true)) if i!=class_id])
        alpha_new=alpha[int(class_id)]
        gamma_new=gamma[int(class_id)]
        class_pred=y_pred[int(class_id)]
        neg_pred=np.sum(y_pred[neg_ids])
        y_true_new=np.array([0,1])
        y_pred_new=np.array([neg_pred,class_pred])
        ones=K.ones_like(y_true_new)
        #alpha为行向量，需要转化为列向量
        alpha_t=y_true_new*alpha_new+(ones-y_true)*(1-alpha_new)
        p_t=y_true_new*y_pred_new+(ones-y_true_new)*(ones-y_pred_new)+K.epsilon()
        multi_focal_loss=-alpha_t*K.pow((ones-p_t).gamma_new)*K.log(p_t)
        return multi_focal_loss
        
        
        
        
        
#多个样本，batchsize>1
def multi_focal_loss(alpha,gamma):
    #注意多分类时，标签使用one-hot，y_true.shape=[batch_size,num_class]
    #可以考虑将[0,0,0,1,0,0,0]处理为[0,1]的二分类形式
    def multi_focal_loss_fixed(y_true,y_pred):
        multi_focal_losses=[]
        for y in range(len(y_true)):
            class_id=np.argmax(y_true[y])
            neg_ids=np.array([int(i) for i in range(len(y_true[y])) if i!=class_id])
            alpha_new=alpha[int(class_id)]
            gamma_new=gamma[int(class_id)]
            class_pred=y_pred[y][int(class_id)]
            neg_pred=np.sum(y_pred[y][neg_ids])
            y_true_new=np.array([0,1])
            y_pred_new=np.array([neg_pred,class_pred])
            ones=K.ones_like(y_true_new)
            #alpha为行向量，需要转化为列向量
            alpha_t=y_true_new*alpha_new+(ones-y_true)*(1-alpha_new)
            p_t=y_true_new*y_pred_new+(ones-y_true_new)*(ones-y_pred_new)+K.epsilon()
            multi_focal_loss=-alpha_t*K.pow((ones-p_t).gamma_new)*K.log(p_t)
            multi_focal_losses.append(multi_focal_loss)
        loss=np.mean(multi_focal_losses)
        return loss
    return multi_focal_loss_fixed
        
        
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
 
    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0)
        y_true = tf.cast(y_true, tf.float32)
 
        loss = -y_true*tf.math.pow(1-y_pred, self.gamma)*tf.math.log(y_pred)
        loss = tf.reduce_sum(loss, axis=1)
        return loss
        
        
        
        
        
class F1Score(tf.keras.metrics.Metric):
  def __init__(self,num_class,steps,name='f1_score',**kwargs):
    super(F1Score,self).__init__()
    self.num_class=num_class
    self.steps=steps
    self.p=self.add_weight(shape=(self.num_class,),name='P',initializer='zeros')
    self.r=self.add_weight(shape=(self.num_class,),name='R',initializer='zeros')
  def update_state(self,y_true,y_pred,sample_weight=None):
    y_true_onehot=tf.one_hot(indices=y_true,depth=self.num_class,on_value=1,off_value=0,axis=-1,dtype=tf.float32)
    y_pred=tf.argmax(y_pred,axis=-1)
    y_pred_onehot=tf.one_hot(indices=y_pred,depth=self.num_class,on_value=1,off_value=0,axis=-1,dtype=tf.float32)
    tp_add_fn=tf.reduce_sum(y_true_onehot,axis=0)
    tp_add_fn=tf.add(tp_add_fn,tf.keras.backend.epsilon())
    tp_add_fp=tf.reduce_sum(y_pred_onehot,axis=0)
    tp_add_fp=tf.add(tp_add_fp,tf.keras.backend.epsilon())
    tp_=tf.reduce_sum(tf.multiply(y_true_onehot,y_pred_onehot),axis=0)
    p_=tf.divide(tp_,tp_add_fp)
    r_=tf.divide(tp_,tp_add_fn)
    self.p.assign_add(p_)
    self.r.assign_add(r_)

  def result(self):
    p_add_r=tf.add(tf.add(self.p,self.r),tf.keras.backend.epsilon())
    f1s=tf.divide(tf.multiply(tf.multiply(self.p,self.r),2),p_add_r)
    f1s=tf.reshape(f1s,(-1))
    f1=tf.reduce_mean(f1s,axis=-1)
    f1=tf.divide(f1,self.steps)
    return f1

  def reset_states(self):
    p_init=tf.zeros(shape=(self.num_class,),dtype=tf.float32)
    r_init=tf.zeros(shape=(self.num_class,),dtype=tf.float32)
    self.p.assign(p_init)
    self.r.assign(r_init)
    
    
#需要注意的是：在多分类上focal_loss的效果不是很稳定
class MultiFocalLoss(tf.keras.losses.Loss):
  #gamma:float为一个数值
  def __init__(self,alpha,gamma,**kwargs):
    super(MultiFocalLoss,self).__init__()
    self.alpha_transpose=tf.reshape(tf.cast(alpha,dtype=tf.float64),(-1,1))
    self.gamma=gamma

  def call(self,y_true,y_pred):
    #直接将alpha乘到最后的loss上，前面不进行loss的计算，相当于对loss进行放缩
    y_true_onehot=tf.one_hot(indices=y_true,depth=self.alpha_transpose.shape[0],on_value=1,off_value=0,axis=-1,dtype=tf.float64)
    #pos_alpha=tf.matmul(y_true_onehot,self.alpha_transpose)
    #ones=tf.ones_like(pos_alpha,dtype=tf.float64)
    #neg_alpha=ones-pos_alpha
    #alpha_t=tf.concat([pos_alpha,neg_alpha],axis=1)
    pos_p=tf.multiply(y_true_onehot,tf.cast(y_pred,dtype=tf.float64))
    pos_p=tf.reduce_sum(pos_p,axis=-1)
    pos_p=tf.reshape(pos_p,(-1,1))
    ones_p=tf.ones_like(pos_p,dtype=tf.float64)
    neg_p=ones_p-pos_p
    p_t=tf.concat([pos_p,neg_p],axis=1)
    ones_pt=tf.ones_like(p_t,dtype=tf.float64)
    #gamma_mat=tf.fill(ones_pt.shape,self.gamma)
    gamma=tf.cast(gamma,dtype=tf.float64)
    loss=-tf.math.pow((ones_pt-p_t),gamma_mat)*tf.math.log(p_t)
    loss=tf.reduce_mean(loss,axis=0)#求一个batch内的loss平均值
    loss=tf.multiply(alpha,loss)
    return loss