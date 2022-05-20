import math
from data_generate import *
import numpy as np
import tensorflow as tf
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os
#去除tf警告
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


FLAGS=tf.flags.FLAGS

tf.flags.DEFINE_string('url_','http://mattmahoney.net/dc/','the front part of data net address')

tf.flags.DEFINE_string('filename','text8.zip','the path of zip file')

tf.flags.DEFINE_integer('batch_size',128,'')

tf.flags.DEFINE_integer('embedding_size',128,'')

tf.flags.DEFINE_integer('skip_window',2,'the window of skip-gram use')

tf.flags.DEFINE_integer('num_skips',4,'the number of sample by each data')

tf.flags.DEFINE_integer('vocab_size',50000,'all vocabularies of the whole table')

tf.flags.DEFINE_integer('valid_size',16,'the valid dataset size')

tf.flags.DEFINE_integer('valid_window',100,'valid will be selected in this and the size equal to valid size')

tf.flags.DEFINE_integer('num_sample',64,'the number of neg sample which is working for nce-loss')

tf.flags.DEFINE_integer('train_steps',5001,'')

def plot_with_labels(low_dim_embs_,labels_,filename='w2v.png'):
    """
    根据降维后的emb向量画出向量分布图
    :param low_dim_embs_: 降维后的向量表示
    :param labels_: 降维后的向量对应的word
    :param filename: 图片保存路径
    :return: None
    """
    assert low_dim_embs_.shape[0]>=len(labels_),"more labels than embedding"
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels_):
        x,y=low_dim_embs_[i,:]
        plt.scatter(x,y)
        #annotate标注文字
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)


def main(_):
    # valid_examples：ids，从0-valid_window里随机选择valid_size个id
    valid_examples = np.random.choice(FLAGS.valid_window, FLAGS.valid_size, replace=False)
    graph=tf.Graph()

    with graph.as_default():
        train_inputs=tf.placeholder(tf.int32,shape=[FLAGS.batch_size])
        train_labels=tf.placeholder(tf.int32,shape=[FLAGS.batch_size,1])
        valid_dataset=tf.constant(valid_examples,dtype=tf.int32)

        with tf.device('/cpu:0'):
            #embedding表示整个词表的初始化权重矩阵
            embedding=tf.Variable(tf.random_normal([FLAGS.vocab_size,FLAGS.embedding_size],-1.0,1.0))
            #embedding_lookup：实际就是根据输入的ids从embedding中取出对应的权重
            embed=tf.nn.embedding_lookup(embedding,train_inputs)

            nce_weight=tf.Variable(tf.truncated_normal([FLAGS.vocab_size,FLAGS.embedding_size],stddev=1.0/math.sqrt(FLAGS.embedding_size)))
            nce_bias=tf.Variable(tf.zeros(FLAGS.vocab_size))

        loss=tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                           biases=nce_bias,
                                           labels=train_labels,
                                           inputs=embed,
                                           num_sampled=FLAGS.num_sample,
                                           num_classes=FLAGS.vocab_size))

        optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        norm=tf.sqrt(tf.reduce_sum(tf.square(embedding),axis=1,keepdims=True))
        norm_emb=embedding/norm
        valid_embedding=tf.nn.embedding_lookup(norm_emb,valid_dataset)
        #similarity计算相似度，选择top-k的词语
        similarity=tf.matmul(valid_embedding,norm_emb,transpose_b=True)

        init=tf.global_variables_initializer()

    words=read_data('text8.zip')
    data,count,dic,inv_dic=build_dataset(word=words,vocab_size=FLAGS.vocab_size)


    with tf.Session(graph=graph) as sess:
        init.run()

        avg_loss=0
        for step in range(FLAGS.train_steps):
            batch_inputs,batch_labels=generate_batch(FLAGS.batch_size,FLAGS.num_skips,FLAGS.skip_window,0,data)
            feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}

            loss_val,_=sess.run([loss,optimizer],feed_dict=feed_dict)
            avg_loss+=loss_val

            if step % 200==0:
                if step>0:
                    avg_loss/=200
                print("avg loss at step %d",step,':',avg_loss)
                avg_loss=0

            if step % 2000==0:
                sim=sess.run(similarity)

                for i in range(FLAGS.valid_size):
                    #从valid_examples中取出对应的id，通过id在inv_dic中找到对应的word
                    valid_word=inv_dic[valid_examples[i]]
                    top_k=8
                    #argsort返回-sim[i,:]从小到大的数的索引值，索引值对应的就是某个word的id
                    nearest=(-sim[i,:]).argsort()[1:top_k+1]
                    log_str='Nearest to %s:' % valid_word

                    for k in range(top_k):
                        near_word=inv_dic[nearest[k]]
                        log_str='%s %s,' % (log_str,near_word)

                    print(log_str)
        #得到归一化后的emb
        final_emb=sess.run(norm_emb)

    tsne=TSNE(init='pca',n_iter=5000)
    plot_cnt=100
    low_dim_embs=tsne.fit_transform(final_emb[:plot_cnt,:])
    labels=[inv_dic[i] for i in range(plot_cnt)]
    plot_with_labels(low_dim_embs,labels)


if __name__=='__main__':
    tf.app.run()