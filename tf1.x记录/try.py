import tensorflow as tf
import os
import numpy as np

x=tf.Variable(1.0)
y=tf.Variable(0.0)
x_add_1=tf.assign_add(x,1)

with tf.control_dependencies([x_add_1]):
    print('a')
    y=tf.identity(x)
init=tf.initialize_variables([x])

with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        print(sess.run(y))



x=tf.compat.v1.placeholder(tf.int32,shape=[None,2])
print(x)



x=tf.Variable(1.0,name='input',trainable=True)
y=tf.Variable(1.0,name='input2',trainable=False)
tf.summary.scalar()
print(x,y)


a=tf.Variable(1.,tf.float32)
b=tf.Variable(2.,tf.float32)
num=10

model_save_path='/home/mi/py36-tf115/try/model'
model_name='model'

saver=tf.compat.v1.train.Saver()

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for step in range(num):
        c=sess.run(tf.add(a,b))
        checkpoint=os.path.join(model_save_path,model_name)
        print(os.path.abspath(checkpoint))
        saver.save(sess,checkpoint,global_step=step)


a=tf.Variable(1.,tf.float32)
b=tf.Variable(2.,tf.float32)
num=10

model_save_path='/home/mi/py36-tf115/try/model'
model_name='model'

saver=tf.compat.v1.train.Saver()

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    ckpt=tf.compat.v1.train.get_checkpoint_state(model_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)


g1=tf.Graph()
print(g1)
with g1.as_default():
    v=tf.get_variable('v',shape=[3],initializer=tf.contrib.layers.xavier_initializer())
    print(v)
g2=tf.Graph()
with g2.as_default():
    v=tf.get_variable('v',shape=[5],initializer=tf.contrib.layers.xavier_initializer())
    print(v)
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('',reuse=True):
        print(sess.run(tf.get_variable('v')))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('',reuse=True):
        print(sess.run(tf.get_variable('v')))