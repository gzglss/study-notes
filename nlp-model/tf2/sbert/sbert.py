# sbert tf2实现
from transformers import BertConfig, TFBertMainLayer, TFBertPreTrainedModel
import tensorflow as tf
import os


class TFMyBertModel(TFBertPreTrainedModel):
    def __init__(self, config, max_len, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.max_len = max_len
        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")
        self.avg = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, **kwargs):
        output = self.bert(inputs, **kwargs)
        sequence = self.avg(output.last_hidden_state)
        return sequence

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
                "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)
        return output


class MyModel(tf.keras.Model):
    def __init__(self,  num_label, model_dir, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        config = BertConfig.from_json_file(os.path.join(model_dir, 'config.json'))
        self.bert = TFMyBertModel.from_pretrained(model_dir+'/tf_model.h5', config=config)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(num_label,
                                           kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                                           use_bias=True,
                                           bias_initializer=tf.zeros_initializer(),
                                           activation=None)

    def call(self, inputs, **kwargs):
        sequence_a = self.bert((inputs['a_input_ids'], inputs['a_attention_mask'], inputs['a_token_type_ids']))
        sequence_b = self.bert((inputs['b_input_ids'], inputs['b_attention_mask'], inputs['b_token_type_ids']))
        sub = tf.abs(tf.subtract(sequence_a, sequence_b))
        concentrate = tf.concat([sequence_a, sequence_b, sub], axis=1)
        concentrate = self.dropout(concentrate)
        logits = self.dense(concentrate)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        a_norm = sequence_a / (tf.sqrt(tf.reduce_sum(tf.square(sequence_a), 1, keepdims=True)) + 1e-8)
        b_norm = sequence_b / (tf.sqrt(tf.reduce_sum(tf.square(sequence_b), 1, keepdims=True)) + 1e-8)
        cos_sim = tf.reduce_sum(tf.multiply(b_norm, a_norm), 1)
        return sequence_a, sequence_b, probabilities, log_probs, cos_sim