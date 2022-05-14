import collections

import tensorflow as tf

FLAGS=tf.flags.FLAGS

tf.flags.DEFINE_string('train_path','data/train.csv','the path of train data')

tf.flags.DEFINE_string('dev_path','data/dev.csv','the path of dev data')

tf.flags.DEFINE_string('output_dir','model/cls_model','the save path of model')

tf.flags.DEFINE_integer('batch_size',32,'the batch size of training')

tf.flags.DEFINE_integer('epoch',10,'the epoch of training')

tf.flags.DEFINE_boolean('do_train',True,'training or not')

tf.flags.DEFINE_boolean('do_eval',True,'eval or not')

#输入导入
def load_data(input_file):
    examples,labels = [],[]
    with tf.io.gfile.GFile(input_file,'r') as infile:
        one=True
        for line in infile.readlines():
            if one:
                one=False
                continue
            line=line.strip().split(',')
            examples.append(line[:-1])
            labels.append(line[-1])
        infile.close()
    return examples,labels

#编写输入函数
def train_input_fn(data,labels,batch_size):
    features=split_data(data)
    print(features)
    labels=list(map(int,labels))
    dataset=tf.data.Dataset.from_tensor_slices((features,labels))
    dataset=dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def split_data(data):
    SepalLength, SepalWidth, PetalLength, PetalWidth = [], [], [], []
    for i in range(len(data)):
        SepalLength.append(float(data[i][0]))
        SepalWidth.append(float(data[i][1]))
        PetalLength.append(float(data[i][2]))
        PetalWidth.append(float(data[i][3]))
    features = dict()
    features['SepalLength'] = SepalLength
    features['SepalWidth'] = SepalWidth
    features['PetalLength'] = PetalLength
    features['PetalWidth'] = PetalWidth
    return features

def create_float_feature(data):
    f=tf.train.Feature(float_list=tf.train.FloatList(value=data))
    return f

def create_int_feature(data):
    f=tf.train.Feature(int64_list=tf.train.Int64List(value=data))
    return f

def eval_input_fn(data,labels,batch_size):
    features=split_data(data)
    if labels is None:
        inputs=features
    else:
        labels = list(map(int, labels))
        inputs=(features,labels)
    dataset=tf.data.Dataset.from_tensor_slices(inputs)
    dataset=dataset.batch(batch_size)
    return dataset

def my_model(features,labels,mode,params):
    net=tf.feature_column.input_layer(features,params['feature_cols'])
    for i in params['hidden_size']:
        net=tf.layers.dense(net,units=i,activation=tf.tanh)

    logits=tf.layers.dense(net,params['n_classes'])
    pred=tf.argmax(logits,axis=1)
    if mode==tf.estimator.ModeKeys.PREDICT:
        predictions={
            'class_ids':pred[:,tf.newaxis],
            'proba':tf.nn.softmax(logits),
            'logits':logits
        }
        return tf.estimator.EstimatorSpec(mode,predictions=predictions)

    loss=tf.losses.sparse_softmax_cross_entropy(labels,logits)

    acc=tf.metrics.accuracy(labels=labels,
                            predictions=pred,
                            name='acc_op')
    metric={'acc':acc}
    eval_log={
        'labels':labels,
        'pred':pred
    }

    # eval_hooks=tf.train.LoggingTensorHook(tensors=eval_log,every_n_iter=1)
    if mode==tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metric)
        # return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metric,evaluation_hooks=[eval_hooks])

    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
        global_step=tf.train.get_global_step()
        train_op=optimizer.minimize(loss,global_step=global_step)

        tvars=tf.trainable_variables()
        for var in tvars:
            tf.logging.info('name=%s,shape=%s',var.name,var.shape)

        train_log={
            'global_step':global_step,
            'acc':acc,
            'loss':loss,
            'labels':labels
        }

        # train_hooks=tf.train.LoggingTensorHook(tensors=train_log,every_n_iter=1)
        # return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op,training_hooks=[train_hooks])
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(_):
    feature_cols=[]
    feat_names=['SepalLength','SepalWidth','PetalLength','PetalWidth']
    for col in feat_names:
        feature_cols.append(tf.feature_column.numeric_column(col))

    tf.logging.info('feature_cols_info:\n%s',feature_cols)

    train_x,train_y=load_data(FLAGS.train_path)
    dev_x,dev_y=load_data(FLAGS.dev_path)
    steps=len(train_y)*FLAGS.epoch
    params={
        'feature_cols':feature_cols,
        'hidden_size':[16,64,16],
        'n_classes':3
    }
    classifier=tf.estimator.Estimator(model_fn=my_model,
                                      model_dir=FLAGS.output_dir,
                                      params=params)

    if FLAGS.do_train:
        classifier.train(input_fn=lambda:train_input_fn(train_x,train_y,FLAGS.batch_size),steps=steps)

    if FLAGS.do_eval:
        eval_res=classifier.evaluate(input_fn=lambda:eval_input_fn(dev_x,dev_y,FLAGS.batch_size))
        tf.logging.info('dev acc:%s',eval_res)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)