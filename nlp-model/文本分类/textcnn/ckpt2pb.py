import tensorflow as tf



checkpointdir="../model/textcnn_model-35000"
# checkpoint=tf.train.latest_checkpoint(checkpointdir)

graph=tf.Graph()
with graph.as_default():
    sess=tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpointdir))
        builder = tf.saved_model.builder.SavedModelBuilder("../model/pbmodel")
        saver.restore(sess, checkpointdir)
        builder.add_meta_graph_and_variables(sess,
                                             ['serve'],
                                             strip_default_attrs=False)
        builder.save()
        # node_names=[n.name for n in tf.get_default_graph().as_graph_def().node]
        # print(node_names)