"""
输入数据-数据处理-转换为input_fn
模型函数（model_fn）
estimator建立模型
训练
验证
预测
模型输出
"""
import tensorflow as tf
from tensorflow.python.estimator.run_config import RunConfig
try:
    from best_exporter import *
    from data_progress import *
    from model_builder import *
except:
    from .data_progress import *
    from .best_exporter import *
    from .model_builder import *

Flags=tf.flags.FLAGS

tf.flags.DEFINE_string('ptm_path','/fds/gzg/pretrain_model','the path of ptm')

tf.flags.DEFINE_string('data_path','/fds/gzg/data/tnews','the path of data')

tf.flags.DEFINE_string('save_path','/fds/gzg/model','the export path of model')

tf.flags.DEFINE_integer("train_batch_size",64,'train batch size')

tf.flags.DEFINE_integer("eval_batch_size",64,'eval batch size')

tf.flags.DEFINE_integer('epochs',3,'training epochs')

tf.flags.DEFINE_integer('num_class',None,'the num of class')

tf.flags.DEFINE_integer('num_filter',8,'the num of filter')

tf.flags.DEFINE_integer('max_seq_len',64,'the max len of seq')

tf.flags.DEFINE_integer('save_checkpoints_steps',1000,'the step for saving checkpoints')

tf.flags.DEFINE_integer('lstm_unit',256,'the unit num of lstm')

tf.flags.DEFINE_integer('dense_unit',512,'the unit num of dense after lstm')

tf.flags.DEFINE_float("init_learning_rate",0.0001,'the init lr')

tf.flags.DEFINE_float("keep_prob",0.7,'the num of unit will be kept')

tf.flags.DEFINE_list('filter_sizes',[3,4,5],'the sizes of filter')

tf.flags.DEFINE_boolean('use_bilstm',True,'whether using bilstm')

tf.flags.DEFINE_boolean('use_textcnn',False,'whether using textcnn')

def construct_estimator(bert_config,
                        init_checkpoint,
                        max_seq_len,
                        num_class,
                        train_steps,
                        warm_up_steps,
                        use_bilstm,
                        use_textcnn,
                        lstm_unit,
                        keep_prob,
                        dense_unit
                        ):
    model_fn=model_build_fn(bert_config_file=bert_config,
                            init_checkpoint=init_checkpoint,
                            max_seq_length=max_seq_len,
                            use_bilstm=use_bilstm,
                            use_textcnn=use_textcnn,
                            )
    run_config=RunConfig(model_dir=Flags.save_path,save_checkpoints_steps=Flags.save_checkpoints_steps)

    tf.logging.info('use normal estimator')
    estimator=tf.estimator.Estimator(model_fn=model_fn,
                                     params={'train_batch_size':Flags.train_batch_size,
                                             'eval_batch_size':Flags.eval_batch_size,
                                             'init_learning_rate':Flags.init_learning_rate,
                                             'num_class':num_class,
                                             'filter_sizes':Flags.filter_sizes,
                                             'num_filter':Flags.num_filter,
                                             'train_steps':train_steps,
                                             'warm_up_steps':warm_up_steps,
                                             'lstm_unit':lstm_unit,
                                             'keep_prob':keep_prob,
                                             'dense_unit':dense_unit},
                                     config=run_config)
    return estimator

def get_eval_result(result,output_file,tokenizer,id2label):
    total,correct=0,0
    # if os.path.isdir(output_file):
    out_dir=os.path.join(output_file,'result_info.txt')
    with tf.gfile.Open(out_dir,'w') as res_file:
        for idx,res in enumerate(result):
            label_pred=res['pred_label']
            label_proba=res['proba']
            truth_label=res['truth_label']
            input_ids=res['input_ids']
            input_mask=res['input_mask']

            valid_input_ids=[input_id for input_id,mask in zip(input_ids,input_mask) if mask==1]
            tokens=tokenizer.convert_ids_to_tokens(valid_input_ids)

            try:
                res_file.write('%s\t%s\t%s\t%s\n' % (' '.join(tokens),id2label[truth_label],id2label[label_pred[0]],label_proba[label_pred[0]]))
            except:
                tf.logging.info("*** truth label is %s ***" % truth_label)
                tf.logging.info('*** the label pred is %s ***' % label_pred)
                tf.logging.info('*** the label proba is %s ***' % label_proba)
                # tf.logging.info("*** the res info ***")
                # tf.logging.info('res: %s' % res)

            if truth_label==label_pred:
                correct+=1
            total+=1
    res_file.close()
    acc=round(correct/total*100,3)
    result={'dev_set_acc':acc}
    return json.dumps(result)


def train(bert_config,
          init_checkpoint,
          output_dir,
          tokenizer,
          summary_writer,
          train_path,
          dev_path,
          labels,
          num_class,
          train_steps,
          warm_up_steps,
          data_len,
          use_bilstm,
          use_textcnn,
          lstm_unit,
          keep_prob,
          dense_unit):
    max_seq_len=Flags.max_seq_len
    train_batch_size=Flags.train_batch_size
    eval_batch_size=Flags.eval_batch_size
    if not os.path.exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    #label与id的转换
    label2id_map={}
    id2label_map={}
    for idx,label in enumerate(labels):
        label2id_map[label]=idx
        id2label_map[idx]=label

    tf.logging.info('*** all labels ***')
    tf.logging.info(' '.join(labels))

    # tf.logging.info('use estimator config')

    #导入数据
    train_data=load_data(train_path,shuffle=True)
    dev_data=load_data(dev_path,shuffle=False)

    #数据包装
    dp=DataProcessor(max_seq_len=Flags.max_seq_len)

    #构造估计器
    estimator=construct_estimator(bert_config=bert_config,
                                  init_checkpoint=init_checkpoint,
                                  max_seq_len=max_seq_len,
                                  num_class=num_class,
                                  train_steps=train_steps,
                                  warm_up_steps=warm_up_steps,
                                  use_bilstm=use_bilstm,
                                  use_textcnn=use_textcnn,
                                  lstm_unit=lstm_unit,
                                  keep_prob=keep_prob,
                                  dense_unit=dense_unit,
                                  )

    tf.logging.info('***** running training *****')
    tf.logging.info('num example=%d' % data_len)
    tf.logging.info('batch size=%d' % train_batch_size)
    tf.logging.info('num steps=%d' % train_steps)

    train_file=os.path.join(output_dir,'train.tf_record')
    #将数据转化为tf_record的形式，并保存在文件
    dp.file_base_convert_examples_to_features(examples=train_data,
                                              label2id_map=label2id_map,
                                              tokenizer=tokenizer,
                                              output_file=train_file)

    #根据文件构造训练集的input_fn
    train_input_fn=file_base_input_fn_builder(input_file=train_file,
                                              max_seq_len=max_seq_len,
                                              is_training=True)

    # dev_input_fn_list=[]
    # for dev_example in dev_data:
    #     dev_input_fn_list.append(input_fn_builder(features=dp.convert_example_to_features(dev_example,label2id_map,tokenizer),
    #                                               is_training=False,
    #                                               max_seq_len=max_seq_len))

    #构造评测集的input_fn
    dev_input_fn=input_fn_builder(features=dp.convert_example_to_features(dev_data,label2id_map,tokenizer),
                                  is_training=False,
                                  max_seq_len=max_seq_len)


    best_ckpt_dir=os.path.join(output_dir,'best_checkpoint')
    tf.gfile.MakeDirs(best_ckpt_dir)

    #搭建best model的输出接受器，用以导入最优参数
    best_ckpt_exporter=BestCheckpointsExporter(
        serving_input_receiver_fn=serving_fn,
        best_checkpoint_path=best_ckpt_dir,
        compare_fn=loss_smaller
    )

    #包装input_fn
    train_spec=tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=train_steps
    )

    #指定测试集最优ckpt输出的位置
    eval_spec=tf.estimator.EvalSpec(
        input_fn=dev_input_fn,
        steps=None,
        start_delay_secs=60,
        throttle_secs=60,
        exporters=best_ckpt_exporter,
    )

    #estimator自带的训练并评测的接口
    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    #读取最优的ckpt的路径
    with tf.gfile.Open(os.path.join(best_ckpt_dir,'best_checkpoint.txt'),'r') as f:
        best_checkpoint=json.loads((f.readline()))['best_checkpoint_path']
    f.close()

    tf.logging.info('***** evaluate on the best checkpoint: %s *****' % best_checkpoint)

    #搭建最优的估计器
    best_estimator=construct_estimator(bert_config,
                                       best_checkpoint,
                                       max_seq_len,
                                       num_class,
                                       train_steps,
                                       warm_up_steps,
                                       use_bilstm=use_bilstm,
                                       use_textcnn=use_textcnn,
                                       lstm_unit=lstm_unit,
                                       keep_prob=keep_prob,
                                       dense_unit=dense_unit,
                                       )

    result=best_estimator.predict(input_fn=dev_input_fn,yield_single_examples=True)
    eval_res=get_eval_result(result=result,output_file=output_dir,tokenizer=tokenizer,id2label=id2label_map)
    summary_writer.write('%s\n' % eval_res)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    labels, data_len = None, None
    num_class = Flags.num_class
    if num_class is None:
        num_class, labels, data_len = get_label(Flags.data_path + 'toutiao_category_train.txt')
    else:
        _, _data_len = get_label(Flags.data_path + 'toutiao_category_train.txt')

    all_data = data_len * Flags.epochs
    train_steps=all_data//Flags.train_batch_size
    if all_data % Flags.train_batch_size!=0:
        train_steps+=1

    # 设置warm_up_step
    warm_up_steps = int(train_steps * 0.1)

    vocab_path=Flags.ptm_path+'vocab.txt'
    bert_config_path=Flags.ptm_path+'bert_config.json'
    init_checkpoint=Flags.ptm_path+'bert_model.ckpt'
    output_dir=Flags.save_path
    tokenizer=FullTokenizer(vocab_path, do_lower_case=True)
    train_path=Flags.data_path+'toutiao_category_train.txt'
    dev_path=Flags.data_path+'toutiao_category_dev.txt'
    summary_file=os.path.join(output_dir,'result_summary.txt')

    lstm_unit=Flags.lstm_unit
    keep_prob=Flags.keep_prob
    dense_unit=Flags.dense_unit

    with tf.gfile.Open(summary_file,'w') as summary:
        train(bert_config=bert_config_path,
              init_checkpoint=init_checkpoint,
              output_dir=output_dir,
              tokenizer=tokenizer,
              summary_writer=summary,
              train_path=train_path,
              dev_path=dev_path,
              labels=labels,
              num_class=num_class,
              train_steps=train_steps,
              warm_up_steps=warm_up_steps,
              data_len=data_len,
              use_bilstm=True,
              use_textcnn=False,
              lstm_unit=lstm_unit,
              keep_prob=keep_prob,
              dense_unit=dense_unit,
              )
        summary.close()

if __name__=='__main__':
    tf.app.run()