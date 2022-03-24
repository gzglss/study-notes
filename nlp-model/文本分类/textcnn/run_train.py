from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
# from tensorflow.contrib import learn
import numpy as np
import random

try:
    import loaddata
    import textcnn
except:
    from . import loaddata
    from . import textcnn

tf.flags.DEFINE_string("traindir", None, "train file")
tf.flags.DEFINE_string("devdir", None, "dev file")
tf.flags.DEFINE_string("vocabdir", None, "vocab file")
tf.flags.DEFINE_string("outdir", None, "out file")

tf.flags.DEFINE_integer("emb_size", 64, "the embedding size of word")
tf.flags.DEFINE_integer("max_len", 32, "max seq len")
tf.flags.DEFINE_list("filter_size", [3, 4, 5], "the size of filter")
tf.flags.DEFINE_integer("num_filter", 64, "the numbers of filter")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "1-dropout_rate")
tf.flags.DEFINE_float("l2_reg", 0.0, "l2 regularization")

tf.flags.DEFINE_integer("batch_size", 256, "batch_size")
tf.flags.DEFINE_integer("epochs", 2, "epochs")
tf.flags.DEFINE_integer("num_checkpoint", 5000, "num_checkpoint to store")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps")
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS

print("================loading data==================")
text, label, vocab_size = loaddata.load_data(FLAGS.traindir, FLAGS.vocabdir,5000000)
dev_text, dev_label, _ = loaddata.load_data(FLAGS.devdir, FLAGS.vocabdir,20000)
print("train len:",len(text))
print("\ndev len:",len(dev_text))
print("================load data end==================")
c = list(zip(text, label))
random.seed(2022)
random.shuffle(c)
text[:], label[:] = zip(*c)


# x = np.zeros((len(text), FLAGS.max_len))
# for i in range(len(text)):
#     for j in range(FLAGS.max_len):
#         x[i][j] = text[i][j]
# print("x_shape:", x.shape)

with tf.Session() as sess:
    cnn = textcnn.TextCnn(seq_len=FLAGS.max_len,
                  num_class=2,
                  vocab_size=vocab_size,
                  emb_size=FLAGS.emb_size,
                  filter_sizes=FLAGS.filter_size,
                  num_filters=FLAGS.num_filter,
                  l2_reg=FLAGS.l2_reg,
                  )

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    print("tf.global_variables:", tf.global_variables())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoint)

    sess.run(tf.global_variables_initializer())


    def train_step(x_batch, y_batch):
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, loss, acc = sess.run(
            [train_op, global_step, cnn.loss, cnn.accuracy], feed_dict
        )
        print("step {}, loss {:g}, acc {:g}".format(step, loss, acc))


    def dev_step(x, y):
        feed_dict = {
            cnn.input_x: x,
            cnn.input_y: y,
            cnn.dropout_keep_prob: 1.0
        }
        step, loss, accuracy = sess.run(
            [global_step, cnn.loss, cnn.accuracy],
            feed_dict)
        print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))


    batches = loaddata.data_generate(
        text, label, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
    print("=======================start training==========================")
    # step = 0
    for batch in batches:
        x_batch, y_batch = batch
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\n======================Evaluation:========================")
            dev_step(dev_text, dev_label)
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, FLAGS.outdir, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
        # print('\r{%d}/{%d}' % (step, len(x) / FLAGS.batch_size * FLAGS.epochs), end='', flush=True)
        # step += 1

# if __name__ == "__main__":
tf.flags.mark_flag_as_required("traindir")
tf.flags.mark_flag_as_required("devdir")
tf.flags.mark_flag_as_required("outdir")
    # tf.app.run()
