import tensorflow as tf
import os
import loaddata
from textcnn import TextCnn
import numpy as np

tf.flags.DEFINE_string("testdir",None, "test file")
tf.flags.DEFINE_string("vocabdir",None, "vocab file")
tf.flags.DEFINE_string("resultdir",None, "result file")
tf.flags.DEFINE_string("checkpointdir", None, "Checkpoint directory from training run")

FLAGS=tf.flags.FLAGS

text=[]
with open(FLAGS.testdir,'r') as tfile:
    for line in tfile.readlines():
        line=line.strip()
        text.append(line)

vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocabdir)
x_test = np.array(list(vocab_processor.transform(text)))

checkpoint=tf.train.latest_checkpoint(FLAGS.checkpointdir)

graph=tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        saver=tf.train.Saver()
        saver.restore(sess, checkpoint)
        input_x=graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob=graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        prediction=graph.get_operation_by_name("predicts").outputs[0]
        preds=sess.run(prediction,feed_dict={input_x:x_test,dropout_keep_prob:1.0})

y=[]
for pred in preds:
    if pred==1:
        y.append("michat")
    else:
        y.append("other")

with open(FLAGS.resultdir,'a') as rfile:
    for i in range(len(y)):
        rfile.write("%s\t%s\n"%(text[i],y[i]))
    rfile.close()

print("=================prediction end======================")