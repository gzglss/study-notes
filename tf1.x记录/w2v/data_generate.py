import collections,math,os,random,zipfile,urllib
import tensorflow as tf
import numpy as np

# url_="http://mattmahoney.net/dc/"

"""
从网址获取数据
"""
def get_url_data(filename,expected_byte,url_):
    #expected_byte文件大小
    if not os.path.exists(filename):
        filename,_=urllib.request.urlretrieve(url_+filename,filename)
    stat_info=os.stat(filename)
    if stat_info.st_size==expected_byte:
        print('found and verified',filename)
    else:
        print(stat_info.st_size)
        raise Exception('failed to verify'+filename+'. can you get to it with a browser')
    return filename

# filename=get_url_data('text8.zip',31344016)

"""
读入zip数据
"""
def read_data(path):
    with zipfile.ZipFile(path) as f:
        print(f.namelist())
        data=tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


"""
创建词汇表：
1.统计词频，取top5w作为vocab
2.创建词:id的哈希表
3.对于未登录词使用[unk]代替，编号为0，并统计数量
"""

def build_dataset(word,vocab_size):
    count_=[['UNK',-1]]
    count_.extend(collections.Counter(word).most_common(vocab_size-1))
    dic_={}
    for w,_ in count_:
        dic_[w]=len(dic_)
    data_=list()
    unk_count=0
    for w in word:
        if w in dic_:
            idx=dic_[w]
        else:
            idx=0
            unk_count+=1
        data_.append(idx)
    count_[0][1]=unk_count
    reverse_dic_=dict(zip(dic_.values(),dic_.keys()))
    return data_,count_,dic_,reverse_dic_

def generate_batch(batch_size,num_skips,skip_window,data_idx,data):
    """
    生成batch数据
    :param data: 单词的id列表
    :param data_idx: 当前单词的id
    :param batch_size: ..,
    :param num_skips: 对每个单词生成的样本数，也就是由这个单词可以产生的样本数量
    :param skip_window: ...
    :return: 一个batch的样本
    """
    assert batch_size % num_skips==0
    assert num_skips <= 2*skip_window
    #batch内包含的数据是目标单词本身+上下文
    batch=np.zeros(shape=(batch_size,),dtype=np.int32)
    #labels只能是上下文，因为需要通过目标单词生成上下文
    labels=np.zeros(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1#一个单词可以影响到span区间
    buffer=collections.deque(maxlen=span)
    #将第一个span用来初始化buffer
    #data表示单词对应的id
    for _ in range(span):
        buffer.append(data[data_idx])
        data_idx += 1
    for i in range(batch_size//num_skips):
        target=skip_window#一个span内，目标单词就是中间的那个
        target_avoid=[target]#存储目标单词，一个span中label不能为其中的单词
        for j in range(num_skips):
            #在span内遍历选择一个目标作为label，同时要保证目标不在禁止列表里
            #因为span的长度比较短，所以可以直接遍历，随机数容易陷入循环
            idx=0
            while target in target_avoid:
                target=idx
                idx+=1
            target_avoid.append(target)#更新禁止列表
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j,0]=buffer[target]
        #由于给deque设置了maxlen，所以这里添加一个数，就会在前面删除一个数
        #相当于把整体区间整体后移了一步
        buffer.append(data[data_idx])
        data_idx+=1
    return batch,labels