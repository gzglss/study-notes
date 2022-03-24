# simcse tf2实现

#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
import tensorflow as tf


def simcse_loss(pooled_output):
    idxs = tf.range(0, tf.shape(pooled_output)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = tf.equal(idxs_1, idxs_2)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.l2_normalize(pooled_output, axis=1)
    similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
    similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
    return tf.reduce_mean(loss)