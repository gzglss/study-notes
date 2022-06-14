import tensorflow as tf

try:
    from modeling import *
    from optimization import *
except:
    from .modeling import *
    from .optimization import *


def model_build_fn(bert_config_file, init_checkpoint, max_seq_length, use_bilstm, use_textcnn):
    def model_fn(features, labels, mode, params):
        """
        后面调用model_fn时，并没有传入参数这一步，网上说features和labels是由input_fn传入的
        :param features: input_fn传入
        :param labels: input_fn传入
        :param mode: estimator来定义
        :param params: 在配置estimator时设置,RunConfig
        :return:
        """
        # if labels is None:
        #     labels=features['label_ids']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # label处理
        if 'label_ids' in features.keys():
            label_ids = features['label_ids']
        else:
            label_ids = None

        bert_config = BertConfig.from_json_file(bert_config_file)
        model, input_ids, input_mask, lengths = build_bert_base_model(bert_config, features, is_training,
                                                                      max_seq_length)

        last_layer_output = model.get_sequence_output()

        # encoder_layers = [model.get_embedding_output()]  # 将bert的emb输出添加到encoder_layers
        # encoder_layers.extend(model.get_all_encoder_layers())  # 将bert的所有输出层添加到列表

        # 这里使用了广播的功能，[batch_size,from_seq_len,1]*[batch_size,1,to_seq_len]->[batch_size,from_seq_len,to_seq_len]
        # 为维度值为1表示，对应维度所有的值都与该值相乘
        # attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

        # 使用bert的第0层作为输入，也就是微调每一层的参数
        # transformer_input = encoder_layers[1]

        # 建立模型，得到模型的输出，只返回最后一层的输出
        # 做一些下游的微调任务，既可以使用cls的输出，也可以使用最后一层的输出
        # 这里相当于在bert和textcnn之间的加了一个12层的transformer，所以不能这么写
        # last_layer_output = transformer_model(
        #     input_tensor=transformer_input,
        #     attention_mask=attention_mask,
        #     hidden_size=768,
        #     num_hidden_layers=12,
        #     intermediate_size=3072,
        #     hidden_dropout_prob=bert_config.hidden_dropout_prob,
        #     attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
        #     initializer_range=bert_config.initializer_range,
        #     do_return_all_layers=False,
        # )

        tvars = tf.trainable_variables()
        # 从bert原始模型中加载初始参数
        tf.logging.info('=========load vars===========')
        tf.logging.info('init_checkpoint:%s' % init_checkpoint)
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info('=========load over===========')

        # tf.logging.info('===========all vars=============')
        # for var in tvars:
        #     tf.logging.info('name=%s, shape=%s' % (var.name, var.shape))
        logits, loss, pred, proba = None, None, None, None
        if use_textcnn:
            params['filter_sizes'] = [3, 4, 5]  # 发现输入得到的不对，这里暂时重新定义
            logits, loss, pred, proba = textcnn_model(last_layer_output, params['num_class'], params['filter_sizes'],
                                                      params['num_filter'], label_ids)
        if use_bilstm:
            loss, pred, proba, logits = bilstm(input_tensor=last_layer_output,
                                               label_ids=label_ids,
                                               lstm_unit=params['lstm_unit'],
                                               keep_prob=params['keep_prob'],
                                               dense_unit=params['dense_unit'],
                                               num_class=params['num_class'])

        if not use_bilstm and not use_textcnn:
            raise AttributeError('you must input a downstream model type')

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = create_optimizer(loss=loss,
                                        init_lr=params['init_learning_rate'],
                                        num_train_steps=params['train_steps'],
                                        num_warmup_steps=params['warm_up_steps'],
                                        )
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            # tf.logging.info('*** the label is %s , and the pred is %s' % (labels,pred))
            print('*** the pred is %s ***' % pred)
            print('*** the label ids is %s ***' % label_ids)
            acc = tf.metrics.accuracy(labels=label_ids, predictions=pred)
            p = tf.metrics.precision(labels=label_ids, predictions=pred)
            r = tf.metrics.recall(labels=label_ids, predictions=pred)
            # metrics里需要传入指标和update_op，如果要把f1加入的话可能要自建op，还不会哦
            # a = tf.constant([2],dtype=tf.float32)
            # p=tf.cast(p,dtype=tf.float32)
            # r=tf.cast(r,dtype=tf.float32)
            # f1 = tf.divide(tf.multiply(a,tf.multiply(p,r)), tf.add(p, r))

            metrics = {
                'acc': acc,
                'p': p,
                'r': r,
            }

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metrics
            )

        else:
            prediction = {'pred_label': pred,
                          'proba': tf.nn.softmax(logits),
                          'logits': logits,
                          'truth_label': label_ids,
                          'input_ids': input_ids,
                          'input_mask': input_mask,
                          }

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=prediction
            )

        return output_spec

    return model_fn


def build_bert_base_model(bert_config, features, is_training, max_seq_length):
    tf.logging.debug("***features***")
    for name in sorted(features.keys()):
        tf.logging.debug('name=%s , shape=%s' % (name, features[name].shape))

    input_ids = features['input_ids']
    lengths = features['length']
    segment_ids = tf.zeros(get_shape_list(input_ids), dtype=tf.int32)

    # 将maxlen大于lengths的部分置0
    input_mask = tf.sequence_mask(lengths, maxlen=max_seq_length, dtype=tf.int32)

    model = BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        token_type_ids=segment_ids,
        input_mask=input_mask
    )

    return model, input_ids, input_mask, lengths
