import tensorflow as tf
from input_fn import create_input_fn
from seq2seq_model import Seq2Seq
import vocab
import dill
import utils
import numpy as np

tf.flags.DEFINE_string("source_file", None, "input file")
tf.flags.DEFINE_string("params_path", None, "param path")
tf.flags.DEFINE_string("model_dir", None, "directory to load model from")
tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")
tf.flags.DEFINE_string("checkpoint_path", None,
                       """Full path to the checkpoint to be loaded. If None,
                       the latest checkpoint in the model dir is used.""")

FLAGS = tf.flags.FLAGS

def params_load(f):
    fp = open(f, 'br')
    return  dill.load(fp)
 
def main(_argv):
    #saver = tf.train.Saver()
    if not FLAGS.source_file:
        raise ValueError("You must specify source_file")

    if not FLAGS.params_path:
        raise ValueError("You must specify params_path")

    if not FLAGS.model_dir:
        raise ValueError("You must specify model_dir")

    hparams = params_load(FLAGS.params_path)
    #checkpoint_path = FLAGS.checkpoint_path
    #if not checkpoint_path:
    #    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    #print(checkpoint_path) 
    predict_input_fn = create_input_fn(source_file_list=[FLAGS.source_file],
                                     target_file_list=None,
                                     batch_size=FLAGS.batch_size,
                                     scope="predict_input_fn")

    source_vocab_info = vocab.get_vocab_info(hparams.source_vocab_path)
    target_vocab_info = vocab.get_vocab_info(hparams.target_vocab_path)
    print(source_vocab_info)

    seq2seq = Seq2Seq(target_vocab_info.special_vocab.SEQUENCE_START,
                      target_vocab_info.special_vocab.SEQUENCE_END,
                      hparams)
 
    estimator = tf.contrib.learn.Estimator(model_fn=seq2seq.create_model_fn(), 
                                           params=hparams, model_dir=FLAGS.model_dir) 

    for preds in estimator.predict(input_fn=predict_input_fn):
        print(utils.to_sent_string(preds['predictions'].tolist()))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
