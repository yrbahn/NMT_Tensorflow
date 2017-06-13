import tensorflow as tf
import numpy as np
import nltk

from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.training import training
from tensorflow.python.estimator.estimator import _check_hooks_type
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.estimator import model_fn as model_fn_lib
from utils import string_decode, to_sent_string

class RLEstimator(tf.estimator.Estimator):
    def __init__(self, model_fn, model_dir=None, config=None, params=None):
        super(RLEstimator, self).__init__(model_fn, model_dir, config, params)
            
    def rl_train(self, input_fn, model, hooks=None, steps=None, max_steps=None):
        if (steps is not None) and (max_steps is not None):
            raise ValueError('Can not provide both steps and max_steps.')
        if steps is not None and steps <= 0:
            raise ValueError('Must specify steps > 0, given: {}'.format(steps))
        if max_steps is not None and max_steps <= 0:
            raise ValueError(
                'Must specify max_steps > 0, given: {}'.format(max_steps))
        if max_steps is not None:
            start_step = _load_global_step_from_checkpoint_dir(self._model_dir)
            if max_steps <= start_step:
                logging.info('Skipping training since max_steps has already saved.')
                return self
            
        hooks = _check_hooks_type(hooks)
        if steps is not None or max_steps is not None:
            hooks.append(training.StopAtStepHook(steps, max_steps))
        loss = self._rl_train_model(input_fn=input_fn, model=model, hooks=hooks)
        logging.info('Loss for final step: %s.', loss)
        return self
                    
    def _rl_train_model(self, input_fn, model, hooks):
        all_hooks = []
        with ops.Graph().as_default() as g, g.device(self._device_fn):
            random_seed.set_random_seed(self._config.tf_random_seed)
            global_step_tensor = training.create_global_step(g)
            with ops.device('/cpu:0'):
                features, labels = input_fn()
            
            estimator_spec = self._call_model_fn(features, labels,
                                                 model_fn_lib.ModeKeys.TRAIN)
            
            ops.add_to_collection(ops.GraphKeys.LOSSES, estimator_spec.loss)
            all_hooks.extend([
                #training.NanTensorHook(estimator_spec.loss),
                #training.LoggingTensorHook(
                #    {
                #        'loss': estimator_spec.loss,
                #        'step': global_step_tensor
                #    },
                #    every_n_iter=100)
            ])
            all_hooks.extend(hooks)
            all_hooks.extend(estimator_spec.training_hooks)
        
            if not (estimator_spec.scaffold.saver or
                ops.get_collection(ops.GraphKeys.SAVERS)):
                ops.add_to_collection(ops.GraphKeys.SAVERS,
                    training.Saver(
                        sharded=True,
                        max_to_keep=self._config.keep_checkpoint_max,
                        defer_build=True))
                
            chief_hooks = []
            if (self._config.save_checkpoints_secs or self._config.save_checkpoints_steps):
                saver_hook_exists = any([
                    isinstance(h, training.CheckpointSaverHook)
                    for h in (all_hooks + chief_hooks +
                        list(estimator_spec.training_chief_hooks))
                ])
                if not saver_hook_exists:
                    chief_hooks = [
                        training.CheckpointSaverHook(
                            self._model_dir,
                            save_secs=self._config.save_checkpoints_secs,
                            save_steps=self._config.save_checkpoints_steps,
                            scaffold=estimator_spec.scaffold)
                    ]
            with training.MonitoredTrainingSession(
                master=self._config.master,
                is_chief=self._config.is_chief,
                checkpoint_dir=self._model_dir,
                scaffold=estimator_spec.scaffold,
                hooks=all_hooks,
                chief_only_hooks=chief_hooks + estimator_spec.training_chief_hooks,
                save_checkpoint_secs=0,  # Saving is handled by a hook.
                save_summaries_steps=self._config.save_summary_steps,
                config=config_pb2.ConfigProto(allow_soft_placement=True)) as mon_sess:
          
                loss = None
                feed_dict = {model.rl_training : True, model.advantage: [0.0]}
                _num = 0
                while not mon_sess.should_stop():
                    #print(feed_dict)
                    _features, _labels = mon_sess.run([features, labels], feed_dict=feed_dict)
                    _feed_dict, _, _   = model.create_feed_dict_from_input_fn(_features, _labels)
                    feed_dict.update(_feed_dict)
                    preds = mon_sess.run(estimator_spec.predictions, feed_dict=feed_dict)
                    advantage = []
                    blue = []
                    for i, label in enumerate(_labels['target_tokens'].tolist()):
                        target_sent   = string_decode(label[1:])
                        sampling_sent = string_decode(preds['sampling_sent'][i].tolist())
                        greedy_sent   = string_decode(preds['greedy_sent'][i].tolist())
                        
                        baseline_blue_score = nltk.translate.bleu_score.sentence_bleu([target_sent], greedy_sent)
                        blue_score = nltk.translate.bleu_score.sentence_bleu([target_sent], sampling_sent)
                        advantage.append(-blue_score + baseline_blue_score)
                        blue.append(baseline_blue_score)
                    
                    feed_dict.update({model.advantage:np.array(advantage)})
                    _, loss, preds = mon_sess.run([estimator_spec.train_op, estimator_spec.loss, estimator_spec.predictions], feed_dict=feed_dict)
                    
                    #print(advantage)
                    if _num % 50 == 0:
                        print('blue:', sum(blue)/len(blue))
                    _num +=1
                    #preds, _, _ = mon_sess.run([estimator_spec.predictions, estimator_spec.loss, estimator_spec.train_op])
                    #print(preds['J'])
                    #print(preds['crossent'])
            return loss
