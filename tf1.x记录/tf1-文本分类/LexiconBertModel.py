"""
给文本分类引入词表信息
包括：LexiconBertConfig、embedding_lookup、embedding_post_procession
其余部分与bert-model没有区别
"""
import copy

import tensorflow as tf
import six
try:
    from modeling import *
except:
    from .modeling import *

class LexiconBertConfig(BertConfig):
    def __init__(self,
                 char_vocab_size,
                 word_vocab_size,
                 additional_vocab_size=0,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        super(LexiconBertConfig, self).__init__(None,
                                                hidden_size,
                                                num_hidden_layers,
                                                num_attention_heads,
                                                intermediate_size,
                                                hidden_act,
                                                hidden_dropout_prob,
                                                attention_probs_dropout_prob,
                                                max_position_embeddings,
                                                type_vocab_size,
                                                initializer_range)
        self.char_vocab_size=char_vocab_size
        self.word_vocab_size=word_vocab_size
        self.additional_vocab_size=additional_vocab_size

    @classmethod
    def from_dict(cls, json_object):
        #重写from_dict方法
        config = LexiconBertConfig(char_vocab_size=0,word_vocab_size=0,additional_vocab_size=0)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        if not config.vocab_size:
            config.vocab_size=config.char_vocab_size+config.word_vocab_size+config.additional_vocab_size
        return config

class LexiconBertModel:
    """
    引入词表信息，提升模型对语义边界的判别能力，从而使其在下游任务上取得更好表现
    引入词表，增强某些实体或句子中的关键词的编码能力，使其对句子的语义起到更大的作用，进而可以减弱非关键词的影响，使编码更能表达句子的核心意思
    """
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_heads,
                 input_tails,
                 emb_init_method='naive',
                 emb_process_method='span_head_tail_avg',
                 input_mask=None,
                 token_type_ids=None,
                 segment_ids=None,
                 use_one_hot_embedding=False,
                 scope=None,
                 ):
        """
        LexiconBert的输入参数，与bert不同的有
        input_heads：词语span的开头token
        input_tails：词语span的结尾token
        emb_init_method：使用Lexicon后，初始化embedding的方法tag，有多种，会在后面的定义中说明
        emb_process_method：使用Lexicon后，后处理embedding的方法tag
        token_type_ids：该token属于词还是字
        segment_ids：文本A还是文本B
        """
        config=copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob=0.
            config.attention_probs_dropout_prob=0.

        input_shape=get_shape_list(input_ids,expected_rank=3)
        batch_size=input_shape[0]
        seq_len=input_shape[1]

        if input_mask is None:
            input_mask=tf.ones(shape=[batch_size,seq_len],dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids=tf.zeros(shape=[batch_size,seq_len],dtype=tf.int32)

        if segment_ids is None:
            segment_ids=tf.zeros(shape=[batch_size,seq_len],dtype=tf.int32)

        with tf.variable_scope(scope,default_name='lexicon-bert'):
            with tf.variable_scope('embedding'):
                (self.embedding,self.embedding_table)=


def embedding_lookup(input_ids,
                     char_vocab_size,
                     word_vocab_size,
                     addition_vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     init_emb_method='naive',
                     word_embedding_name='word_embeddings',
                     use_one_hot_embedding=False):
    """
    根据输入句子的token ids，查表找到对应的embedding vec
    init_emb_method：初始化emb的方式
    naive：简单的初始化方法，直接将char和word拼接
    head_tail_avg：将char保持不变，word的head和tail取平均
    head_tail_combine：char保持不变，word的head和tail引入两个参数进行combine
    span_avg：char不变，对整个word span取平均
    span_combine：char不变，对整个word span进行combine
    span_addition_avg：引入addition
    span_addition_combine：引入addition
    """
    #当input_ids的shape为[batch_size,seq_len]时，给其增加一维
    if input_ids.shape.ndims==2:
        input_ids=tf.expand_dims(input_ids,-1)

    if init_emb_method=='naive':
        vocab_size=char_vocab_size+word_vocab_size
        embedding_table=tf.get_variable(name=word_embedding_name,
                                        shape=[vocab_size,embedding_size],
                                        initializer=create_initializer(initializer_range))

        flat_input_ids=tf.reshape(input_ids,shape=[-1])
        if use_one_hot_embedding:
            one_hot_input_ids=tf.one_hot(flat_input_ids,dtype=vocab_size)
            output=tf.matmul(one_hot_input_ids,embedding_table)
        else:
            output=tf.gather(embedding_table,flat_input_ids)

        input_shape=get_shape_list(input_ids)
