import time,reader,collections
import numpy as np
import tensorflow as tf

def generate_vocab(input_path):
    counter=collections.Counter()
    with open(input_path,'r') as infile:
        for line in infile.readlines():
            for w in line.strip().split():
                counter[w]+=1
        infile.close()
    sort_word=sorted(counter.items(),key=lambda x:x[1],reverse=True)
    sort_word=['<eos>']+[x[0] for x in sort_word]
    return sort_word

def words_to_ids(vocab):
    word2id={}
    for i,w in enumerate(vocab):
        word2id[w]=i
    return word2id

def data_to_ids(path,vocab):
    word2id=words_to_ids(vocab)
    data_ids=[]
    with open(path,'r') as file:
        for line in file.readlines():
            line=line.strip().split()+['<eos>']
            # ids=[]
            for i in line:
                if i in word2id:
                    data_ids.append(int(word2id[i]))
                else:
                    data_ids.append(int(word2id['<unk>']))
            # line=' '.join(ids)
            # data_ids.append(ids)
    return data_ids

def ptb_producer(data,batch_size,num_steps,name):
    data=tf.convert_to_tensor(data,name="data",dtype=tf.int32)
    data_len=tf.size(data)
    batch_len=data_len//batch_size
    data=tf.reshape(data[:batch_size*batch_len],[batch_size,batch_len])
    epoch_size=(batch_len-1)//num_steps
    assertion=tf.assert_positive(epoch_size,message="epoch_size==0,error")
    with tf.control_dependencies([assertion]):
        epoch_size=tf.identity(epoch_size,name="epoch_size")
    i=tf.train.range_input_producer(epoch_size,shuffle=False).dequeue()
    x=tf.strided_slice(data,[0,i*num_steps],[batch_size,(i+1)*num_steps])
    x.set_shape([batch_size,num_steps])
    y=tf.strided_slice(data,[0,i*num_steps+1],[batch_size,(i+1)*num_steps+1])
    y.set_shape([batch_size,num_steps])
    return x,y

def ptb_raw_data(path):
    train_path=path+'ptb.train.txt'
    valid_path=path+'ptb.valid.txt'
    test_path=path+'ptb.test.txt'

    vocab=generate_vocab(train_path)

    train_data=data_to_ids(train_path,vocab)
    valid_data=data_to_ids(valid_path,vocab)
    test_data=data_to_ids(test_path,vocab)
    return train_data,valid_data,test_data,len(vocab)


class PTBInput(object):
    """
    处理PTB数据
    """
    def __init__(self,config,data,name=None):
        """
        num_steps:LSTM的展开步数,应该是一个句子中单词的个数
        epoch_size:每个epoch需要的迭代数
        """
        self.batch_size=batch_size=config.batch_size
        self.num_steps=num_steps=config.num_steps
        self.epoch_size=((len(data)//batch_size)-1)//num_steps
        self.data=data
        self.input_data,self.targets=ptb_producer(data,batch_size,num_steps,name=name)