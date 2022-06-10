import os
import time
import json
import glob
import shutil
import tensorflow as tf

class BestCheckpointsExporter(tf.estimator.BestExporter):
    def __init__(self,
                 name='best_exporter',
                 serving_input_receiver_fn=None,
                 event_file_pattern='eval/*.tfevents.*',
                 compare_fn=None,
                 assets_extra=None,
                 as_text=False,
                 exports_to_keep=5,
                 best_checkpoint_path=None):
        tf.estimator.BestExporter.__init__(self,name,serving_input_receiver_fn,event_file_pattern,compare_fn,as_text,assets_extra,exports_to_keep)
        self.best_checkpoint_path=best_checkpoint_path

    def export(self,estimator,export_path,checkpoint_path,eval_result,is_the_final_export):
        if self._best_eval_result is None or self._compare_fn(self._best_eval_result,eval_result):
            tf.logging.info('exporting a better model ({} instead of {})...'.format(eval_result,self._best_eval_result))

            best_ckpt_path=os.path.join(self.best_checkpoint_path,str(int(time.time())))
            tf.gfile.MakeDirs(best_ckpt_path)

            #这一句才是将模型输出到best_checkpoint
            #原理是将name文件复制一份后，给到os.path.join(best_ckpt_path,os.path.basename(name))
            for name in glob.glob(checkpoint_path+'.*'):
                shutil.copy(name,os.path.join(best_ckpt_path,os.path.basename(name)))

            with open(os.path.join(self.best_checkpoint_path,'best_checkpoint.txt'),'w') as f:
                content={
                    'best_checkpoint_path':os.path.join(best_ckpt_path,os.path.basename(checkpoint_path))
                }
                f.write('%s' % json.dumps(content,ensure_ascii=False))

            self._best_eval_result=eval_result

            self._garbage_collect_exports(self.best_checkpoint_path)

        else:
            tf.logging.info('keeping the current best model ({} instead of {})'.format(self._best_eval_result,eval_result))


def serving_fn():
    input_ids=tf.placeholder(tf.int32,[None,None],name='input_ids')
    lengths=tf.placeholder(tf.int32,[None],name='length')

    input_fn=tf.estimator.export.build_raw_serving_input_receiver_fn({
        'input_ids':input_ids,
        'length':lengths
    })()

    return input_fn

def loss_smaller(best_eval_result, current_eval_result):
    default_key = "loss"
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')

    return best_eval_result[default_key] > current_eval_result[default_key]