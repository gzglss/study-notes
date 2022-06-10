import tensorflow as tf
import random

try:
    from tokenization import *
except:
    from .tokenization import *

def train_input_fn(feature,labels,batch_size):
    if type(feature)!=dict:
        feature=dict(feature)
    dataset=tf.data.Dataset.from_tensor_slices((feature,labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)

def feature_columns(train_x):
    my_feature_columns=[]
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    return my_feature_columns

def load_data(path,shuffle,batch_size=None,batch=False):
    """
    从文件读取数据，转化为feature后返回
    :param path:
    :param batch_size:
    :param shuffle:
    :param batch: bool，是否以batch返回
    :return:
    """
    with tf.gfile.GFile(path,'r') as f:
        examples,flag=[],True
        for line in f.readlines():
            line=line.strip().split('_!_')
            if line[-1]=='':
                line=line[:-1]
            text,label=line[3],line[2]
            label=label.split('_')[-1]
            assert valid_label(label) is True
            text=convert_to_unicode(text)
            examples.append(Example(text,label))
            if flag:
                tf.logging.info('*** one example from load data ***')
                tf.logging.info('text:%s\t label:%s' % (text,label))
                flag=False

    if shuffle:
        random.shuffle(examples)

    # if batch:
    #     cnt=0
    #     batch_example=[]
    #     while cnt<len(examples):
    #         batch_example.append(examples[cnt])
    #         cnt+=1
    #         if len(batch_example)==batch_size or cnt==len(examples)-1:
    #             yield batch_example
    #             batch_example=[]
    # else:
    return examples

def valid_label(label):
    #保证label为要求的英文
    for w in label:
        if not ('a'<=w<='z'):
            return False
    return True

def get_label(path):
    labels = set()
    cnt=0
    flag_=True
    with tf.gfile.GFile(path,'r') as f:
        for line in f.readlines():
            line = line.strip().split('_!_')
            if flag_:
                tf.logging.info('*** one example from get label')
                tf.logging.info('line:%s\t label:%s' % (line,line[2]))
                flag_=False
            label=line[2]
            label=label.split('_')[-1]
            assert valid_label(label) is True
            labels.add(label)
            cnt+=1
        f.close()
    return len(labels),list(labels),cnt

def file_base_input_fn_builder(input_file,max_seq_len,is_training):
    #定义一个解析字典
    name_to_features={
        'input_ids':tf.FixedLenFeature([max_seq_len],dtype=tf.int64),
        'label_ids':tf.FixedLenFeature([],dtype=tf.int64),
        'length':tf.FixedLenFeature([],dtype=tf.int64)
    }

    def _decode_record(record):
        # print("*** record example ***")
        # print(record)
        #用来解析record文件的数据
        example=tf.parse_single_example(record,name_to_features)

        #转换数据格式，保证GPU和TPU能够正常调用
        for name in list(example.keys()):
            t=example[name]
            if t.dtype==tf.int64:
                t=tf.to_int32(t)
            example[name]=t
        return example

    def input_fn(params):
        tf.logging.info('*** params ***')
        tf.logging.info(params)
        batch_size=params['eval_batch_size']
        data=tf.data.TFRecordDataset(input_file)
        drop_remainder=False
        if is_training:
            batch_size=params['train_batch_size']
            data=data.shuffle(buffer_size=1000)
            # data=data.batch(batch_size)
            drop_remainder=True
        # data=data.batch(batch_size).map(_decode_record,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record),
                batch_size=batch_size,
                drop_remainder=drop_remainder,
                # num_parallel_batches=num_cpu_threads,
            ))
        return data

    return input_fn

def input_fn_builder(features,is_training,max_seq_len):
    assert len(features)>0
    all_input_ids=[]
    all_label_ids=[]
    all_lengths=[]
    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_label_ids.append(feature.label_ids)
        all_lengths.append(feature.length)

    def input_fn(params):
        batch_size=params["train_batch_size"] if is_training else params["eval_batch_size"]
        num_example=len(features)
        feature_map={
            'input_ids':tf.constant(all_input_ids,shape=[num_example,max_seq_len],dtype=tf.int32),
            'length':tf.constant(all_lengths,shape=[num_example],dtype=tf.int32)
        }
        # tf.logging.info('for the input_fn , the label ids is %s' % feature.label_ids)
        if feature.label_ids:
            feature_map['label_ids']=tf.constant(all_label_ids,shape=[num_example],dtype=tf.int32)

        data=tf.data.Dataset.from_tensor_slices(feature_map)
        data=data.batch(batch_size=batch_size)
        # tf.logging.info("the return of input_fn is %s" % data)
        return data

    return input_fn

class Example:
    def __init__(self,text,label):
        self.text=text
        self.label=label

class DataFeature:
    """
    用来将需要的数据包装为feature
    """
    def __init__(self,input_ids,label_ids,length):
        self.input_ids=input_ids
        self.label_ids=label_ids
        self.length=length

class DataProcessor:
    def __init__(self,max_seq_len):
        self.max_seq_len=max_seq_len

    def convert_single_example(self,idx,example,tokenizer,label_map):
        #将单条数据转化为特征
        tokens=[token for token in example.text]
        tokens=["[CLS]"]+tokens+["[SEP]"]
        length=len(tokens)

        input_ids=tokenizer.convert_tokens_to_ids(tokens)

        if length<self.max_seq_len:
            input_ids+=[0]*(self.max_seq_len-length)
        else:
            input_ids=input_ids[:self.max_seq_len]

        assert len(input_ids)==self.max_seq_len

        label=example.label#一个样本
        label_ids=None
        #我们的任务为分类任务
        if label:
            label_ids=label_map[label]
            # label_ids=[label_map[i] for i in labels]

        if idx<3:
            tf.logging.info('*** example ***')
            tf.logging.info('tokens:%s' % ' '.join([printable_text(x) for x in tokenizer.convert_ids_to_tokens(input_ids)]))
            tf.logging.info('input_ids:%s' % ' '.join([str(x) for x in input_ids]))
            tf.logging.info('actual length:%s' % length)

            if label:
                tf.logging.info('labels:%s' % label)
                tf.logging.info('label_ids:%s' % label_ids)
                # tf.logging.info('labels:%s' % ' '.join(labels))
                # tf.logging.info('label_ids:%s' % ' '.join(label_ids))

        # input_ids=tf.constant(input_ids,dtype=tf.int64)
        # label_ids = tf.constant(label_ids, dtype=tf.int64)
        feature=DataFeature(input_ids=input_ids,
                            label_ids=label_ids,
                            length=length)
        return feature

    def convert_example_to_features(self,examples,label2id_map,tokenizer):
        #将数据转化为特征，当数据量较小的时候使用，因为需要把转换后的数据放到CPU的缓存
        features=[]
        for idx,example in enumerate(examples):
            feature=self.convert_single_example(idx=idx,example=example,tokenizer=tokenizer,label_map=label2id_map)
            if feature is None:
                continue
            features.append(feature)
        return features

    def file_base_convert_examples_to_features(self,examples,label2id_map,tokenizer,output_file):
        #将数据转化为features后，保存在tf_record文件中
        tf.logging.info("*** starting convert data to tf_record ***")
        writer=tf.io.TFRecordWriter(output_file)
        for (idx,example) in enumerate(examples):
            if idx==0:
                tf.logging.info('*** data example ***')
                tf.logging.info(example)
            if idx % 10000==0:
                tf.logging.info('writing examples %d of %d' % (idx+1,len(examples)))

            feature=self.convert_single_example(idx=idx,example=example,tokenizer=tokenizer,label_map=label2id_map)
            if feature is None:
                continue

            def create_int_feature(values):
                f=tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features=collections.OrderedDict()
            features['input_ids']=create_int_feature(feature.input_ids)
            features['label_ids']=create_int_feature([feature.label_ids])
            features['length']=create_int_feature([feature.length])

            tf_example=tf.train.Example(features=tf.train.Features(feature=features))
            if idx==0:
                tf.logging.info('*** tf example ***')
                tf.logging.info(tf_example.SerializeToString())
            writer.write(tf_example.SerializeToString())