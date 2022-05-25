import tensorflow as tf

class PTBModel(object):
    def __init__(self,is_training,config,input_):
        self._input=input_
        batch_size=config.batch_size
        num_steps=config.num_steps
        hidden_size=config.hidden_size
        vocab_size=config.vocab_size

        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size,forget_bias=0,state_is_tuple=True)

        attn_cell=lstm_cell#只将attn_cell指向lstm_cell函数名
        if is_training and config.keep_prob<1.0:
            def attn_cell():
                return tf.nn.rnn_cell.DropoutWrapper(lstm_cell(),output_keep_prob=config.keep_prob)
        #将多个rnn cell连接
        cell=tf.nn.rnn_cell.MultiRNNCell([attn_cell() for _ in range(config.num_layers)],state_is_tuple=True)
        self._initial_state=cell.zero_state(batch_size,tf.float32)#初始化

        with tf.device('/cpu:0'):
            embedding=tf.get_variable('embedding',shape=[vocab_size,hidden_size],dtype=tf.float32)
            inputs=tf.nn.embedding_lookup(embedding,input_.input_data)

        if is_training and config.keep_prob<1.0:
            inputs=tf.nn.dropout(inputs,keep_prob=config.keep_prob)

        outputs=[]
        state=self._initial_state

        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step>0:
                    #当不是第一个单词时，变量可以复用
                    tf.get_variable_scope().reuse_variables()
                cell_output,state=cell(inputs[:,time_step,:],state)
                outputs.append(cell_output)#num_steps*[batch_size,hidden_size]

        #num_steps*[batch_size,hidden_size]->[batch_size,num_steps*hidden_size]->[num_steps*batch_size,hidden_size]
        output=tf.reshape(tf.concat(outputs,axis=1),[-1,hidden_size])#在output的第一维进行拼接，相当于将一个句子的不同timestep进行拼接
        #极端情况，将一个句子的每个字的向量表示进行拼接
        softmax_w=tf.get_variable("softmax_w",shape=[hidden_size,vocab_size],dtype=tf.float32)
        softmax_b=tf.get_variable("softmax_b",shape=[vocab_size],dtype=tf.float32)

        logits=tf.matmul(output,softmax_w)+softmax_b
        loss=tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[tf.reshape(input_.targets,[-1])],[tf.ones([batch_size*num_steps],dtype=tf.float32)])
        self._cost=cost=tf.reduce_sum(loss)/batch_size
        self._final_state=state

        if not is_training:
            return

        self._lr=tf.Variable(0.0,trainable=True)
        tvars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),config.max_grad_norm)
        optimizer=tf.train.GradientDescentOptimizer(self._lr)
        self._train_op=optimizer.apply_gradients(zip(grads,tvars),global_step=tf.train.get_global_step())

        self._new_lr=tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update=tf.assign(self._lr,self._new_lr)

    def assign_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self._new_lr:lr_value})

    # 将返回变量设为只读，防止修改变量引发的问题，因为用户无法传入参数，所以无法修改
    #也就是将此方法变成私有属性，在调用时可以不加括号
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    init_scale=0.1
    learning_rate=1.0
    max_grad_norm=5
    num_layers=2
    num_steps=20
    hidden_size=200
    init_lr_epoch=1#初始学习率可训练的epoch数
    max_epoch=2#总共可训练的epoch数
    keep_prob=1.0
    lr_decay=0.5
    batch_size=20
    vocab_size=10000

class TestConfig(object):
    init_scale=0.1
    learning_rate=1.0
    max_grad_norm=1
    num_layers=2
    num_steps=10
    hidden_state=100
    init_lr_epoch=1
    max_epoch=3
    keep_prob=0.9
    lr_decay=0.5
    batch_size=20
    vocab_size=10000


class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000