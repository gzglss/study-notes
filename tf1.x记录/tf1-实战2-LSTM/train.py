import time
import numpy as np
import tensorflow as tf
from data_progress import *
from mymodel import *

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string('data_path','simple-examples/data/','the path of data')

def train(session,model,eval_op=None,verbose=False):
    """
    :param session:sess
    :param model: model
    :param eval_op:
    :param verbose:
    :return:
    """
    start_time=time.time()
    costs=0.0
    iters=0
    state=session.run(model.initial_state)

    fetches={
        "cost":model.cost,
        "final_state":model.final_state
    }

    if eval_op is not None:
        fetches["eval_op"]=eval_op

    print("===start training===")

    for step in range(model.input.epoch_size):
        feed_dict={}
        for i,(c,h) in enumerate(model.initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h

        vals=session.run(fetches,feed_dict)
        cost=vals['cost']
        state=vals['final_state']

        costs+=cost
        iters+=model.input.num_steps

        if verbose and step % (model.input.epoch_size//10)==0:
            print("%.3f perplexity: %.3f speed: %.0f wps:" % (step/model.input.epoch_size,np.exp(costs/iters),iters*model.input.batch_size/(time.time()-start_time)))

    return np.exp(costs/iters)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    raw_data=ptb_raw_data(FLAGS.data_path)
    train_data,valid_data,test_data,_=raw_data
    config=SmallConfig()
    eval_config=SmallConfig()

    graph=tf.Graph()
    with graph.as_default():
        initializer=tf.random_uniform_initializer(-config.init_scale,config.init_scale)

        with tf.variable_scope("Train"):
            train_input=PTBInput(config=config,data=train_data,name="TrainInput")
            with tf.variable_scope("Model",reuse=None,initializer=initializer):
                m=PTBModel(is_training=True,config=config,input_=train_input)

        with tf.variable_scope("valid"):
            valid_input=PTBInput(config=config,data=valid_data,name="ValidInput")
            with tf.variable_scope("Model",reuse=None,initializer=initializer):
                m_valid=PTBModel(is_training=False,config=config,input_=valid_input)

        with tf.variable_scope("Test"):
            test_input=PTBInput(config=config,data=test_data,name='TestInput')
            with tf.variable_scope("Model",reuse=None,initializer=initializer):
                m_test=PTBModel(is_training=False,config=config,input_=test_input)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for i in range(config.max_epoch):
            lr_decay=config.lr_decay**max(i+1-config.init_lr_epoch,0.0)
            m.assign_lr(sess,config.learning_rate*lr_decay)

            print("epoch: %d lr: %.3f" % (i+1,sess.run(m.lr)))

            train_perplexity=train(sess,model=m,eval_op=m.train_op,verbose=True)
            print("epoch: %d train perplexity: %.3f" % (i+1,train_perplexity))

            valid_perplexity=train(sess,m_valid)
            print("epoch: %d valid perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity=train(sess,m_test)
        print("epoch: %d test perplexity: %.3f" % (i + 1, test_perplexity))


if __name__=="__main__":
    tf.app.run()