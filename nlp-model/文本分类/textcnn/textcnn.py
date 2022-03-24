import tensorflow as tf

class TextCnn:
    def __init__(
        self,
        seq_len,
        num_class,
        vocab_size,
        emb_size,
        filter_sizes,
        num_filters,
        l2_reg=0.,
        lr=1e-3
    ):
        self.input_x=tf.placeholder(tf.int32,[None,seq_len],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,num_class],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_out")
        
        l2_loss=tf.constant(0.0)
        
        with tf.name_scope("embedding"):
            self.w=tf.Variable(
                tf.random_uniform([vocab_size,emb_size],-1.0,-1.0),name="w"
            )
            self.embedded_chars=tf.nn.embedding_lookup(self.w,self.input_x)
            self.embedded_chars_expand=tf.expand_dims(self.embedded_chars,-1)
            
        pooled_output=[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv_maxpool-%s"%filter_size):
                filter_shape=[filter_size,emb_size,1,num_filters]
                w=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="w")
                b=tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b")
                
                conv=tf.nn.conv2d(
                    self.embedded_chars_expand,w,strides=[1,1,1,1],padding="VALID",name="conv"
                )
                
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                
                pooled=tf.nn.max_pool(h,ksize=[1,seq_len-filter_size+1,1,1],strides=[1,1,1,1],padding="VALID",name="pool")
                pooled_output.append(pooled)
                
        num_filters_total=num_filters*len(filter_sizes)
        self.h_pool=tf.concat(pooled_output,3)
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])
        
        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)
            
        with tf.name_scope("output"):
            w=tf.get_variable("w",
                              shape=[num_filters_total,num_class],
                              initializer=tf.contrib.layers.xavier_initializer())
            b=tf.Variable(tf.constant(0.1,shape=[num_class]),name="b")
            
            l2_loss+=tf.nn.l2_loss(w)
            l2_loss+=tf.nn.l2_loss(b)
            
            self.scores=tf.nn.xw_plus_b(self.h_drop,w,b,name="scores")
            self.predictions=tf.argmax(self.scores,1,name="predicts")
            
        with tf.name_scope("loss"):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg*l2_loss

        # with tf.name_scope("optimize"):
        #     self.optim=tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
            
        with tf.name_scope("accuracy"):
            correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy")