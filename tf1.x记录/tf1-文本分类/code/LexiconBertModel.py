"""
给文本分类引入词表信息
包括：LexiconBertConfig、embedding_lookup、embedding_post_procession
其余部分与bert-model没有区别
"""

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
        super(LexiconBertConfig, self).__init__(
            None,
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
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.additional_vocab_size = additional_vocab_size

    @classmethod
    def from_dict(cls, json_object):
        # 重写from_dict方法
        config = LexiconBertConfig(char_vocab_size=0, word_vocab_size=0, additional_vocab_size=0)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        if not config.vocab_size:
            config.vocab_size = config.char_vocab_size + config.word_vocab_size + config.additional_vocab_size
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
                 input_pos=None,
                 input_heads=None,
                 input_tails=None,
                 emb_init_method='whole',
                 emb_process_method='naive',
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
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.
            config.attention_probs_dropout_prob = 0.

        input_shape = get_shape_list(input_ids, expected_rank=[2,3])
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_len], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_len], dtype=tf.int32)

        if segment_ids is None:
            segment_ids = tf.zeros(shape=[batch_size, seq_len], dtype=tf.int32)

        with tf.variable_scope(scope, default_name='lexicon-bert'):
            with tf.variable_scope('embedding'):
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    char_vocab_size=config.char_vocab_size,
                    word_vocab_size=config.word_vocab_size,
                    addition_vocab_size=config.additional_vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    emb_init_method=emb_init_method,
                    use_one_hot_embedding=use_one_hot_embedding
                )

                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    input_pos=input_pos,
                    input_heads=input_heads,
                    input_tails=input_tails,
                    emb_process_method=emb_process_method,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    use_segment_ids=True,
                    segment_ids=segment_ids,
                    use_position_embedding=True,
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

                with tf.variable_scope('encoder'):
                    attention_mask = create_attention_mask_from_input_mask(self.embedding_output, input_mask)

                    self.all_encoder_layers = transformer_model(
                        input_tensor=self.embedding_output,
                        attention_mask=attention_mask,
                        hidden_size=config.hidden_size,
                        num_hidden_layers=config.num_hidden_layers,
                        num_attention_heads=config.num_attention_heads,
                        intermediate_size=config.intermediate_size,
                        intermediate_act_fn=get_activation(config.hidden_act),
                        hidden_dropout_prob=config.hidden_dropout_prob,
                        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                        do_return_all_layers=True)

                self.sequence_output = self.all_encoder_layers[-1]

                with tf.variable_scope('pooler'):
                    cls_encode = self.all_encoder_layers[0]
                    # 使用cls做下游任务时，添加了一些非线性
                    self.pooled_output = tf.layers.dense(
                        cls_encode,
                        config.hidden_size,
                        activation=tf.nn.tanh,
                        kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


def get_variable_from_file(input_file, name, expected_shape, dtype):
    """
    当embedding向量需要保存复用时，可以从文件中读取出对应的变量
    """
    with tf.gfile.GFile(input_file, 'r') as f:
        line = f.readline().strip()
        data = json.load(line)  # load：输入为文件，如果line为目标字符串，则需要使用loads
        value = np.array(data['embeddings'])
        assert value.shape == expected_shape
        f.close()

    value = tf.cast(value, dtype=dtype)
    return tf.get_variable(name=name,
                           dtype=dtype,
                           initializer=value)


def embedding_lookup(input_ids,
                     char_vocab_size,
                     word_vocab_size,
                     addition_vocab_size,
                     embedding_size=768,
                     initializer_range=0.02,
                     emb_init_method='whole',
                     embedding_name='lexicon_embeddings',
                     use_one_hot_embedding=False):
    """
    根据输入句子的token ids，查表找到对应的embedding vec
    init_emb_method：初始化emb的方式
    whole：直接将char和word合并为vocab后，求embedding_table
    char_word：分开求char和word的embedding_table后，concat
    char_word_addition：分开求char和word的embedding_table后，再求addition，然后concat

    更具体的：
    获取token embedding，按照init_method有不同的初始化方案，其中
    当init_method为 init_whole_vocab_from_file 时，将从指定embedding_file文件中读取整个embedding table（包含char与word）

    当init_method为 init_whole_vocab 时，若ckpt中存在该变量，将从ckpt中加载，否则将重新初始化

    当init_method为 init_char_word_vocab_from_file 时，将从指定char_embedding_file与word_embedding文件中分别
    读取char embedding table与word embedding table，再经过concat成为 embedding table

    当init_method为 init_char_word_vocab 时，若ckpt中存在该变量，将从ckpt中加载，否则将重新初始化

    当init_method为 init_with_additional_vocab_from_file 时，将在 init_char_word_vocab 的基础上，从指定的
    additional_embedding_file 中加载additional embedding table；该场景适用于新增了额外的词表

    当init_method为 init_with_additional_vocab 时，将从 ckpt 中分别加载 char embedding table, word embedding table 与
    additional embedding table，经concat成为 embedding table；该场景适用于新增词表已经过训练，需要在下游任务继续训练
    """
    # 当input_ids的shape为[batch_size,seq_len]时，给其增加一维
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, -1)

    vocab_size = char_vocab_size + word_vocab_size
    if emb_init_method == 'whole':
        embedding_table = tf.get_variable(
            name=embedding_name + '_naive',
            shape=[vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))

    elif emb_init_method == 'char_word':
        embedding_table_char = tf.get_variable(
            name=embedding_name + '_char',
            shape=[char_vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))
        embedding_table_word = tf.get_variable(
            name=embedding_name + '_word',
            shape=[word_vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))
        embedding_table = tf.concat([embedding_table_char, embedding_table_word], axis=0)

    elif emb_init_method == 'char_word_addition':
        assert addition_vocab_size > 0
        embedding_table_char = tf.get_variable(
            name=embedding_name + '_char',
            shape=[char_vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))
        embedding_table_word = tf.get_variable(
            name=embedding_name + '_word',
            shape=[word_vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))
        embedding_table_addition = tf.get_variable(
            name=embedding_name + '_addition',
            shape=[addition_vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))
        embedding_table = tf.concat([embedding_table_char, embedding_table_word, embedding_table_addition], axis=0)

    else:
        raise ValueError('you have input a wrong init_emb_method %s' % emb_init_method)

    flat_input_ids = tf.reshape(input_ids, shape=[-1])
    if use_one_hot_embedding:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)
    input_shape=get_shape_list(input_ids)
    output = tf.reshape(output, shape=input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return output, embedding_table


def embedding_postprocessor(input_tensor,
                            input_pos=None,
                            input_heads=None,
                            input_tails=None,
                            emb_process_method='naive',
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=2,
                            token_type_embedding_name='token_type_embedding',
                            use_segment_ids=False,
                            segment_ids=None,
                            segment_size=2,
                            segment_embedding_name='segment_embedding',
                            use_position_embedding=True,
                            position_embedding_name='position_embeddings',
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            ):
    """
    input_pos：[batch_size,seq_len]，must be defined when emb_process_method is naive
    input_heads:[batch_size,seq_len],must be defined when emb_process_method is span_*
    input_tails:[batch_size,seq_len],must be defined when emb_process_method is span_*
    input_
    引入lexicon后，需要根据输入的embedding后处理方式对char和word的embedding进行相应的处理
    token_type_ids：表示token属于char(0)还是word(1)
    segment_ids：表示token属于句子A还是句子B
    emb_process_method有:
    naive：表示使用简单的char和word的拼接
    span_head_tail_avg：char不变，word的头尾取平均
    span_head_tail_combine：char不变，引入参数，对头尾取插值
    span_all_avg：对span取平均
    span_all_combine：对word整个取线性变换后的值

    更具体的：
    embedding后处理，在token embedding基础上添加position embedding，token type embedding与segment embedding，其中
    position embedding包含输入序列中位置信息，使模型获知顺序关系
    token type embedding包含输入序列中字词信息，使模型获知当前位置是字或词
    segment embedding包含输入序列片段信息，使模型获知当前处于哪个片段（如语义相似度任务中包含两个片段）

    model version 目前支持 naive，span_head_tail_avg, span_head_tail_combine, span_pos_avg
    主要区别为序列中每个位置的position如何指定，其中

    naive的position由input_pos指定，char与position对应，而word将通过复制若干次使其与占据的position对应，例如
    输入的text为 我在小米公司工作 其中包含（小米， 公司， 工作），模型接收的输入为
    我 在 小 米 公 司 工 作 小米 小米 公司 公司 工作 工作  （input tokens）
    0  1  2  3  4  5  6  7  2    3    4    5    6    7     （input pos）

    span_head_tail_avg与span_head_tail_combine的position由对应span的首尾位置确定，对char而言首尾相同，例如
    输入的text为 我在小米公司工作 其中包含（小米， 公司， 工作），模型接收的输入为
    我 在 小 米 公 司 工 作 小米 公司 工作  （input tokens）
    0  1  2  3  4  5  6  7  2    4    6     （input heads）
    0  1  2  3  4  5  6  7  3    5    7     （input tails）
    在span_head_tail_avg中，每个位置的position embedding = (PE(head) + PE(tail)) / 2
    在span_head_tail_combine中，每个位置的
    position embedding = (p * PE(head) + q * PE(tail)) / 2，其中p, q可训练

    span_pos_avg与span_pos_combine的position由对应span占据的positions确定，对char而言占据单个position，
    对word而言占据多个positions，例如
    输入的text为 我在小米公司工作 其中包含（小米， 公司， 工作），模型接收的输入为
    我 在 小 米 公 司 工 作 小米 公司 工作  （input tokens）
    0  1  2  3  4  5  6  7  2    4    6     （input heads）
    0  1  2  3  4  5  6  7  3    5    7     （input tails）
    在span_pos_avg中，每个位置的position embedding = sum(PE(p in span)) / len(span)
    在span_pos_combine中，每个位置的position embedding = (w1 * PE(p1) + w2 * PE(p2) + ... + wn * PE(pn)) / len(span)
    """
    input_shape = get_shape_list(input_tensor)
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    embedding_size = input_shape[2]

    output = input_tensor
    if use_token_type:
        # 表示是否为char or word，所以这里的token_type_vocab_size固定为2
        assert token_type_ids is not None
        token_type_table = tf.get_variable(name=token_type_embedding_name,
                                           shape=[token_type_vocab_size, embedding_size],
                                           initializer=create_initializer(initializer_range))

        # token_type_vocab_size比较小，所以默认使用one-hot，这样可以获得更快的速度
        flat_token_type_ids = tf.reshape(token_type_ids, shape=[-1])
        one_hot_type_token_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embedding = tf.matmul(one_hot_type_token_ids, token_type_table)
        token_type_embedding = tf.reshape(token_type_embedding, shape=[batch_size, seq_len, embedding_size])
        output += token_type_embedding

    if use_segment_ids:
        # 表示是否使用segment ids来区别输入的句子A、句子B
        if segment_ids is None:
            segment_ids = tf.zeros(shape=[batch_size, seq_len])  # 默认输入一个句子
        segment_table = tf.get_variable(name=segment_embedding_name,
                                        shape=[segment_size, embedding_size],
                                        initializer=create_initializer(initializer_range))
        flat_segment_ids = tf.reshape(segment_ids, shape=[-1])
        one_hot_segment_ids = tf.one_hot(flat_segment_ids, depth=segment_size)
        segment_embedding = tf.matmul(one_hot_segment_ids, segment_table)
        segment_embedding = tf.reshape(segment_embedding, shape=[batch_size, seq_len, embedding_size])
        output += segment_embedding

    if use_position_embedding:
        if input_tails is not None:
            assert_op = tf.assert_less_equal(tf.reduce_max(input_tensor=input_tails), max_position_embeddings)
        elif input_pos is not None:
            assert_op = tf.assert_less_equal(tf.reduce_max(input_tensor=input_pos), max_position_embeddings)
        else:
            raise ValueError('input_pos and input_tail can not both be None')
        with tf.control_dependencies([assert_op]):
            full_position_embedding = tf.get_variable(name=position_embedding_name,
                                                      shape=[max_position_embeddings, embedding_size],
                                                      initializer=create_initializer(initializer_range))

            # position_embedding = tf.slice(full_position_embedding, begin=[0, 0], size=[seq_len, embedding_size])
            # num_dims = output.shape.ndims
            if emb_process_method == 'naive':
                # simple concat the char and word pos emb
                tf.logging.info('use the naive to make a embedding post process')
                # embedding_lookup与tf.gather没太大区别
                # It is a generalization of `tf.gather`
                position_embedding = tf.nn.embedding_lookup(full_position_embedding, input_pos)

            elif emb_process_method == 'span_head_tail_avg':
                tf.logging.info('use the span_head_tail_avg to make a embedding post process')
                head_position_embedding = tf.nn.embedding_lookup(full_position_embedding, input_heads)
                tail_position_embedding = tf.nn.embedding_lookup(full_position_embedding, input_tails)
                position_embedding = (head_position_embedding + tail_position_embedding) / 2

            elif emb_process_method == 'span_head_tail_combine':
                tf.logging.info('use the span_head_tail_combine to make a embedding post process')
                head_position_embedding = tf.nn.embedding_lookup(full_position_embedding, input_heads)
                tail_position_embedding = tf.nn.embedding_lookup(full_position_embedding, input_tails)

                #[2,batch_size*seq_len*emb_size]
                stack_position_embedding=tf.reshape(tf.stack(values=[head_position_embedding,tail_position_embedding],axis=0),shape=[2,-1])
                #这里是将head或tail各自乘以一个相同的权重
                #也就是说，head中的每个val都乘以了相同的权重
                #这里应该也可以增加参数量，使用对head中每个token使用不同的权重，此时weight的shape=[batch_size*seq_len,2]
                #但是，由于这里只是head和tail，所以长度不是seq_len，而是[batch_size*len(input_head),2]
                head_tail_weight = tf.get_variable(
                    name='head_tail_weight',
                    shape=[1, 2],
                    initializer=tf.ones_initializer()
                )
                position_embedding = (tf.matmul(head_tail_weight,stack_position_embedding)) / 2
                position_embedding = tf.reshape(position_embedding,shape=[batch_size,seq_len,embedding_size])

            elif emb_process_method == 'span_all_avg':
                tf.logging.info('use the span_all_avg to make a embedding post process')
                assert tf.less_equal(input_heads, input_tails)
                span_size = input_tails - input_heads + 1
                # span_size=tf.tile(tf.expand_dims(span_size,axis=-1),multiples=[1,1,embedding_size])
                span_pos = get_span_pos(input_heads, input_tails,
                                        max_position_embeddings)  # [batch_size,seq_len,max_position]
                span_position_embedding = tf.nn.embedding_lookup(full_position_embedding,
                                                                 span_pos)  # [batch_size,seq_len,embedding_size]
                position_embedding = tf.reduce_sum(span_position_embedding, axis=-1, keepdims=False)
                position_embedding = tf.div(position_embedding, span_size)

            elif emb_process_method == 'span_all_combine':
                tf.logging.info('use the span_all_combine to make a embedding post process')
                assert tf.less_equal(input_heads, input_tails)
                span_size = input_tails - input_heads + 1
                span_pos = get_span_pos(input_heads, input_tails, max_position_embeddings)
                span_position_embedding = tf.nn.embedding_lookup(full_position_embedding, span_pos)
                w = tf.get_variable(name='span_w',
                                    shape=[seq_len, seq_len],
                                    initializer=create_initializer(initializer_range))
                position_embedding = tf.div(tf.matmul(span_position_embedding, w), span_size)

            else:
                raise ValueError('you have input a wrong emb_post_process %s' % emb_process_method)

            output += position_embedding
    output = layer_norm_and_dropout(output, dropout_prob=dropout_prob)
    return output


def get_span_pos(input_heads, input_tails, max_position):
    """
    根据已有的input_heads和input_tails，得到整个span的position_ids
    其中，padding部分用max_position代替
    input:
        input_heads:[batch_size,seq_len]
        input_tails:[batch_size,seq_len]
    return:
        span_pos_ids=[batch_size,seq_len,max_position]
        eg:[[[mp,mp,...,6,7,8,...,mp,mp],...],...]
    """
    input_shape = get_shape_list(input_heads)
    batch_size = input_shape[0]
    seq_len = input_shape[1]

    input_heads = tf.tile(tf.expand_dims(input_heads, axis=-1),
                          multiples=[1, 1, max_position])  # [batch_size,seq_len,max_position]
    input_tails = tf.tile(tf.expand_dims(input_tails, axis=-1), multiples=[1, 1, max_position])

    #tile与expand_dim交替使用，广播数组，也可以用repeat完成
    span_pos_init = tf.tile(
        tf.expand_dims(
            tf.tile(
                tf.expand_dims(
                    tf.range(start=0, limit=max_position, dtype=tf.int32)
                    , axis=0)
                , multiples=[seq_len, 1])
            , axis=0),
        multiples=[batch_size, 1, 1])

    padding_matrix = tf.ones_like(span_pos_init) * max_position

    # 从span_pos_init中取出对应的span_pos，对于非span部分，直接赋予最大值
    span_pos = tf.where(condition=tf.logical_and(tf.less_equal(span_pos_init, input_tails),
                                                 tf.greater_equal(span_pos_init, input_heads)),
                        x=span_pos_init,
                        y=padding_matrix)
    return span_pos
