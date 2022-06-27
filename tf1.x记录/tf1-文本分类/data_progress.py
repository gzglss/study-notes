import tensorflow as tf
import random

try:
    from tokenization import *
except:
    from .tokenization import *

from minlptokenizer.tokenizer import MiNLPTokenizer

segment_tokenizer=MiNLPTokenizer()

CLS=[101]
SEP=[102]
CHAR_VOCAB_SIZE=8107

def train_input_fn(feature, labels, batch_size):
    if type(feature) != dict:
        feature = dict(feature)
    dataset = tf.data.Dataset.from_tensor_slices((feature, labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


def feature_columns(train_x):
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    return my_feature_columns


def load_data(path, shuffle, batch_size=None, batch=False):
    """
    从文件读取数据，转化为feature后返回
    :param path:
    :param batch_size:
    :param shuffle:
    :param batch: bool，是否以batch返回
    :return:
    """
    with tf.gfile.GFile(path, 'r') as f:
        examples, flag = [], True
        for line in f.readlines():
            line = line.strip().split('_!_')
            if line[-1] == '':
                line = line[:-1]
            text, label = line[3], line[2]
            label = label.split('_')[-1]
            assert valid_label(label) is True
            # text = convert_to_unicode(text)
            examples.append(Example(text, label))
            if flag:
                tf.logging.info('*** one example from load data ***')
                tf.logging.info('text:%s\t label:%s' % (text, label))
                flag = False

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
    # 保证label为要求的英文
    for w in label:
        if not ('a' <= w <= 'z'):
            return False
    return True


def get_label(path):
    labels = set()
    cnt = 0
    flag_ = True
    with tf.gfile.GFile(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('_!_')
            if flag_:
                tf.logging.info('*** one example from get label')
                tf.logging.info('line:%s\t label:%s' % (line, line[2]))
                flag_ = False
            label = line[2]
            label = label.split('_')[-1]
            assert valid_label(label) is True
            labels.add(label)
            cnt += 1
        f.close()
    return len(labels), list(labels), cnt


def file_base_input_fn_builder(input_file, max_seq_len, is_training,use_lexicon):
    # 定义一个解析字典
    if use_lexicon:
        name_to_features = {
            'input_ids': tf.FixedLenFeature([max_seq_len], dtype=tf.int64),
            'input_heads': tf.FixedLenFeature([max_seq_len], dtype=tf.int64),
            'input_tails': tf.FixedLenFeature([max_seq_len], dtype=tf.int64),
            'label_ids': tf.FixedLenFeature([], dtype=tf.int64),
            'length': tf.FixedLenFeature([], dtype=tf.int64)
        }
    else:
        name_to_features = {
            'input_ids': tf.FixedLenFeature([max_seq_len], dtype=tf.int64),
            'label_ids': tf.FixedLenFeature([], dtype=tf.int64),
            'length': tf.FixedLenFeature([], dtype=tf.int64)
        }

    def _decode_record(record):
        # print("*** record example ***")
        # print(record)
        # 用来解析record文件的数据
        example = tf.parse_single_example(record, name_to_features)

        # 转换数据格式，保证GPU和TPU能够正常调用
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        tf.logging.info('*** params ***')
        tf.logging.info(params)
        batch_size = params['eval_batch_size']
        data = tf.data.TFRecordDataset(input_file)
        drop_remainder = False
        if is_training:
            batch_size = params['train_batch_size']
            data = data.repeat()
            data = data.shuffle(buffer_size=1000)
            # data=data.batch(batch_size)
            drop_remainder = True
        # data=data.batch(batch_size).map(_decode_record,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # data = data.apply(
        #     tf.contrib.data.map_and_batch(
        #         lambda record: _decode_record(record),
        #         batch_size=batch_size,
        #         drop_remainder=drop_remainder,
        #         # num_parallel_batches=num_cpu_threads,
        #     ))
        # data = tf.data.Dataset.from_tensor_slices(data)
        data = data.map(_decode_record,num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .batch(batch_size=batch_size,drop_remainder=drop_remainder)
        return data

    return input_fn


def input_fn_builder(features, is_training, max_seq_len,use_lexicon):
    assert len(features) > 0
    all_input_ids = []
    all_label_ids = []
    all_lengths = []
    if use_lexicon:
        all_input_heads = []
        all_input_tails = []
        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_heads.append(feature.input_heads)
            all_input_tails.append(feature.input_tails)
            all_label_ids.append(feature.label_ids)
            all_lengths.append(feature.length)
    else:
        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_label_ids.append(feature.label_ids)
            all_lengths.append(feature.length)

    def input_fn(params):
        batch_size = params["train_batch_size"] if is_training else params["eval_batch_size"]
        num_example = len(features)
        if use_lexicon:
            feature_map = {
                'input_ids': tf.constant(all_input_ids, shape=[num_example, max_seq_len], dtype=tf.int32),
                'input_heads': tf.constant(all_input_heads, shape=[num_example, max_seq_len], dtype=tf.int32),
                'input_tails': tf.constant(all_input_tails, shape=[num_example, max_seq_len], dtype=tf.int32),
                'length': tf.constant(all_lengths, shape=[num_example], dtype=tf.int32)
            }
        else:
            feature_map = {
                'input_ids': tf.constant(all_input_ids, shape=[num_example, max_seq_len], dtype=tf.int32),
                'length': tf.constant(all_lengths, shape=[num_example], dtype=tf.int32)
            }
        # tf.logging.info('for the input_fn , the label ids is %s' % feature.label_ids)
        if feature.label_ids is not None:
            feature_map['label_ids'] = tf.constant(all_label_ids, shape=[num_example], dtype=tf.int32)

        data = tf.data.Dataset.from_tensor_slices(feature_map)
        data = data.batch(batch_size=batch_size)
        # tf.logging.info("the return of input_fn is %s" % data)
        return data

    return input_fn


class Example:
    def __init__(self, text, label):
        self.text = text
        self.label = label


class DataFeature:
    """
    用来将需要的数据包装为feature
    """

    def __init__(self, input_ids, label_ids, length):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.length = length


class DataProcessor:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def convert_single_example(self, idx, example, tokenizer, label_map):
        # 将单条数据转化为特征
        tokens = [token for token in example.text]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        length = len(tokens)
        if length>self.max_seq_len:
            length=self.max_seq_len

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if length < self.max_seq_len:
            input_ids += [0] * (self.max_seq_len - length)
        else:
            input_ids = input_ids[:self.max_seq_len]

        assert len(input_ids) == self.max_seq_len

        label = example.label  # 一个样本
        label_ids = None
        # 我们的任务为分类任务
        if label:
            label_ids = label_map[label]
            # label_ids=[label_map[i] for i in labels]

        if idx < 1:
            tf.logging.info('*** example ***')
            tf.logging.info('example：%s' % example.text)
            tf.logging.info(
                'tokens:%s' % ' '.join([printable_text(x) for x in tokens]))
            tf.logging.info('input_ids:%s' % ' '.join([str(x) for x in input_ids]))
            tf.logging.info('actual length:%s' % length)

            # if label:
            #     tf.logging.info('labels:%s' % label)
            #     tf.logging.info('label_ids:%s' % label_ids)
            #     tf.logging.info('labels:%s' % ' '.join(labels))
            #     tf.logging.info('label_ids:%s' % ' '.join(label_ids))

        # input_ids=tf.constant(input_ids,dtype=tf.int64)
        # label_ids = tf.constant(label_ids, dtype=tf.int64)
        feature = DataFeature(input_ids=input_ids,
                              label_ids=label_ids,
                              length=length)
        return feature

    def convert_example_to_features(self, examples, label2id_map, tokenizer):
        # 将数据转化为特征，当数据量较小的时候使用，因为需要把转换后的数据放到CPU的缓存
        features = []
        for idx, example in enumerate(examples):
            feature = self.convert_single_example(idx=idx, example=example, tokenizer=tokenizer, label_map=label2id_map)
            if feature is None:
                continue
            features.append(feature)
        return features

    def file_base_convert_examples_to_features(self, examples, label2id_map, tokenizer, output_file):
        # 将数据转化为features后，保存在tf_record文件中
        tf.logging.info("*** starting convert data to tf_record ***")
        writer = tf.io.TFRecordWriter(output_file)
        for (idx, example) in enumerate(examples):
            if idx == 0:
                tf.logging.info('*** data example ***')
                tf.logging.info(example)
            if idx % 10000 == 0:
                tf.logging.info('writing examples %d of %d' % (idx + 1, len(examples)))

            feature = self.convert_single_example(idx=idx, example=example, tokenizer=tokenizer, label_map=label2id_map)
            if feature is None:
                continue

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features['input_ids'] = create_int_feature(feature.input_ids)
            features['label_ids'] = create_int_feature([feature.label_ids])
            features['length'] = create_int_feature([feature.length])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            if idx == 0:
                tf.logging.info('*** tf example ***')
                tf.logging.info(tf_example.SerializeToString())
            writer.write(tf_example.SerializeToString())


class LexiconDataFeature:
    """
    用来将需要的数据包装为feature
    """

    def __init__(self, input_ids, input_heads, input_tails, label_ids, length):
        self.input_ids = input_ids
        self.input_heads = input_heads
        self.input_tails = input_tails
        self.label_ids = label_ids
        self.length = length


class LexiconDataProcessor:
    def __init__(self, max_seq_len,lexicon_input):
        self.max_seq_len = max_seq_len
        self.lexicon_input=lexicon_input

    def convert_single_example(self, idx, example, tokenizer, label_map):
        # 将单条数据转化为特征
        segments=custom_segment(example.text,segment_tokenizer)
        if segments is None:
            return None
        tokens = [token for token in example.text]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        char_ids = tokenizer.convert_tokens_to_ids(tokens)
        word_ids,word_heads,word_tails=[],[],[]
        pos=0
        for segment in segments:
            if segment in tokenizer.vocab:
                word_ids.append(tokenizer.vocab[segment])
                word_heads.append(pos)
                word_tails.append(pos+len(segment)-1)
            pos+=len(segment)

        # word_ids：segment对应的ids
        # word_heads：word head对应的pos（也就是索引）
        # word_tails：word tail对应的pos（也就是索引）

        segments_feature={
            'char_ids':char_ids,
            'word_ids':word_ids,
            'word_heads':word_heads,
            'word_tails':word_tails
        }

        #接下来，需要实现word token与word ids的对应
        #对于tagging任务还需要判断分词得到的segment的tag是否合法，必须保证符合词语的tag
        #由于本次的目标只是分类，所以暂时不做处理

        char_ids=CLS+char_ids+SEP
        word_ids=segments_feature['word_ids']
        word_heads=[p+1 for p in segments_feature['word_heads']]#因为最前面加上了CLS，所以需要+1
        word_tails=[p+1 for p in segments_feature['word_tails']]

        input_ids,input_heads,input_tails=[],[],[]

        if self.lexicon_input=='word':
            pos=0
            while pos<len(char_ids):
                if pos in word_heads:
                    #当该pos出现在heads时，表明该位置满足word的要求
                    #根据pos得到对应的索引
                    idx=word_heads.index(pos)
                    input_ids.append(word_ids[idx])#word_ids[idx]：返回pos对应的词的id，也就是在分词后的列表中，该词对应的token id
                    input_heads.append(pos)
                    input_tails.append(word_tails[idx])#该词对应的tails
                    pos+=word_tails[idx]-pos+1
                else:
                    #该pos不在heads中出现，那么表示该pos处的token为char
                    input_ids.append(char_ids[pos])
                    input_heads.append(pos)
                    input_tails.append(pos)
                    pos+=1

        elif self.lexicon_input=='char':
            pos=0#表示索引
            while pos<len(char_ids):
                input_ids.append(char_ids[pos])#char_ids[pos]：pos处对应的token id，也就是说input_ids会存放token ids
                input_heads.append(pos)#那么input_heads应该也是存放token ids才对吧？
                input_tails.append(pos)
                pos+=1

        elif self.lexicon_input=='char_and_word':
            raise NotImplementedError('no support at now')

        else:
            raise ValueError('you have input a wrong lexicon input %s' % self.lexicon_input)

        input_ids=truncate_and_pad(input_ids,self.max_seq_len)
        input_heads=truncate_and_pad(input_heads,self.max_seq_len)
        input_tails=truncate_and_pad(input_tails,self.max_seq_len)

        assert len(input_ids)==len(input_heads)==len(input_tails) == self.max_seq_len

        # tokens=tokenizer.convert_ids_to_tokens(input_ids[1:-1])
        # try:
        #     assert ''.join(tokens)==example.text
        # except:
        #     tf.logging.error('reconstruct sequence error')
        #     tf.logging.error('raw text: %s' % example.text)
        #     tf.logging.error('reconstruct text: %s' % ''.join(tokens))

        label = example.label  # 一个样本
        label_ids = None
        # 我们的任务为分类任务
        if label:
            if type(label)==list:
                label_ids = [label_map[l] for l in example.label]
            else:
                label_ids=label_map[label]

        length=len(tokens)

        if idx < 1:
            tf.logging.info('*** example ***')
            tf.logging.info(
                'tokens:%s' % '|'.join([printable_text(x) for x in tokens]))
            tf.logging.info('input_ids:%s' % ' '.join([str(x) for x in input_ids]))
            tf.logging.info('actual length:%s' % length)

            if label:
                tf.logging.info('labels:%s' % label)
                tf.logging.info('label_ids:%s' % label_ids)
                # tf.logging.info('labels:%s' % ' '.join(labels))
                # tf.logging.info('label_ids:%s' % ' '.join(label_ids))

        # input_ids=tf.constant(input_ids,dtype=tf.int64)
        # label_ids = tf.constant(label_ids, dtype=tf.int64)
        feature = LexiconDataFeature(
            input_ids=input_ids,
            input_heads=input_heads,
            input_tails=input_tails,
            label_ids=label_ids,
            length=length)
        return feature

    def convert_example_to_features(self, examples, label2id_map, tokenizer):
        # 将数据转化为特征，当数据量较小的时候使用，因为需要把转换后的数据放到CPU的缓存
        features = []
        for idx, example in enumerate(examples):
            feature = self.convert_single_example(idx=idx, example=example, tokenizer=tokenizer, label_map=label2id_map)
            if feature is None:
                continue
            features.append(feature)
        return features

    def file_base_convert_examples_to_features(self, examples, label2id_map, tokenizer, output_file):
        # 将数据转化为features后，保存在tf_record文件中
        tf.logging.info("*** starting convert data to tf_record ***")
        writer = tf.io.TFRecordWriter(output_file)
        for (idx, example) in enumerate(examples):
            if idx == 0:
                tf.logging.info('*** data example ***')
                tf.logging.info(example)
            if idx % 10000 == 0:
                tf.logging.info('writing examples %d of %d' % (idx + 1, len(examples)))

            feature = self.convert_single_example(idx=idx, example=example, tokenizer=tokenizer, label_map=label2id_map)
            if feature is None:
                continue

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features['input_ids'] = create_int_feature(feature.input_ids)
            features['input_heads'] = create_int_feature(feature.input_heads)
            features['input_tails'] = create_int_feature(feature.input_tails)
            features['label_ids'] = create_int_feature([feature.label_ids])
            features['length'] = create_int_feature([feature.length])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

def custom_segment(query,tokenizer):
    #使用自定义的分词器进行分词
    if tokenizer:
        query=tokenizer.cut(query)
    else:
        #如果未检测到分词器就只能以char进行训练
        query=[c for c in query]
    return query

def truncate_and_pad(input_ids,max_seq_len):
    if len(input_ids)>=max_seq_len:
        return input_ids[:max_seq_len]
    else:
        input_ids=input_ids+[0]*(max_seq_len-len(input_ids))
        return input_ids