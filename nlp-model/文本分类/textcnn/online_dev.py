import tensorflow as tf
import os
try:
    import loaddata
    import textcnn
except:
    from . import loaddata
    from . import textcnn
import numpy as np

tf.flags.DEFINE_string("devdir",None, "dev file")
tf.flags.DEFINE_string("vocabdir",None, "vocab file")
tf.flags.DEFINE_string("checkpointdir", None, "Checkpoint directory from training run")

FLAGS=tf.flags.FLAGS

dev_text, dev_label, _ = loaddata.load_data(FLAGS.devdir, FLAGS.vocabdir,20000)
dev_y=[0]*len(dev_text)
for i in range(len(dev_text)):
    dev_y[i]=np.argmax(np.array(dev_label[i]))


checkpoint=tf.train.latest_checkpoint(FLAGS.checkpointdir)

graph=tf.Graph()
with graph.as_default():
    sess=tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint))
        saver.restore(sess, checkpoint)
        input_x=graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob=graph.get_operation_by_name("dropout_keep_out").outputs[0]
        prediction=graph.get_operation_by_name("output/predicts").outputs[0]
        preds=sess.run(prediction,feed_dict={input_x:dev_text,dropout_keep_prob:1.0})

print("true:",dev_y)
print("pred:",preds)
c=0
for i in range(len(preds)):
    if preds[i]==dev_y[i]:
        c+=1
print("acc:",c/len(preds))

tf.flags.mark_flag_as_required("devdir")
tf.flags.mark_flag_as_required("vocabdir")
tf.flags.mark_flag_as_required("checkpointdir")