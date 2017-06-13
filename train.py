import tensorflow as tf
from collections import namedtuple
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config

from input_fn import create_input_fn
from seq2seq_model import Seq2Seq
from text_metric_specs import bleu_fn

import tempfile
import vocab

tf.logging.set_verbosity(tf.logging.INFO)

# data
tf.flags.DEFINE_string("source_files", None,
                       """source files for training""")
tf.flags.DEFINE_string("target_files", None,
                       """target files for training""")
tf.flags.DEFINE_string("dev_source_files", None,
                     """source files for evaluting""")
tf.flags.DEFINE_string("dev_target_files", None,
                     """target files for evaluting""")
tf.flags.DEFINE_integer("batch_size", 32, """batch_size""")

tf.flags.DEFINE_string("output_dir", None,
                       """The directory to write model checkpoints and summaries
                       to. If None, a local temporary directory is created.""")

tf.flags.DEFINE_string("optimizer", "Adam", "optimizer")
tf.flags.DEFINE_float("optimizer_clip_gradients", 10.0, "clip_normal")
tf.flags.DEFINE_boolean("rl_training", False, "enable RL training")
tf.flags.DEFINE_string("cell_model", "LSTM", "cell model")
tf.flags.DEFINE_integer("num_layers", 1, """number of layers""")
tf.flags.DEFINE_boolean("attention", False, """enable attention""")
tf.flags.DEFINE_string("source_vocab_path", None, """source vocab path""")
tf.flags.DEFINE_string("target_vocab_path", None, """target vocab path""")
tf.flags.DEFINE_integer("eval_batch_size", 1, """evaluation batch size""")
tf.flags.DEFINE_integer("embedding_dim", 64, """embedding dimension""")
tf.flags.DEFINE_integer("hidden_size", 256, """LSTM hidden size""")
tf.flags.DEFINE_integer("max_source_len", 40, "source input length""")
tf.flags.DEFINE_integer("max_target_len", 40, "target input length""")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")    

# Training parameters
tf.flags.DEFINE_string("schedule", "train",
                       """Estimator function to call, defaults to
                       continuous_train_and_eval for local run""")
tf.flags.DEFINE_integer("train_steps", None,
                        """Maximum number of training steps to run.
                         If None, train forever.""")
tf.flags.DEFINE_integer("eval_every_n_steps", 1000,
                        "Run evaluation on validation data every N steps.")

# RunConfig Flags
tf.flags.DEFINE_integer("tf_random_seed", None,
                        """Random seed for TensorFlow initializers. Setting
                        this value allows consistency between reruns.""")
tf.flags.DEFINE_integer("save_checkpoints_secs", None,
                        """Save checkpoints every this many seconds.
                        Can not be specified with save_checkpoints_steps.""")
tf.flags.DEFINE_integer("save_checkpoints_steps", None,
                        """Save checkpoints every this many steps.
                        Can not be specified with save_checkpoints_secs.""")
tf.flags.DEFINE_integer("keep_checkpoint_max", 5,
                        """Maximum number of recent checkpoint files to keep.
                        As new files are created, older files are deleted.
                        If None or 0, all checkpoint files are kept.""")
tf.flags.DEFINE_integer("keep_checkpoint_every_n_hours", 4,
                        """In addition to keeping the most recent checkpoint
                        files, keep one checkpoint file for every N hours of
                        training.""")
tf.flags.DEFINE_float("gpu_memory_fraction", 1.0,
                      """Fraction of GPU memory used by the process on
                      each GPU uniformly on the same machine.""")
tf.flags.DEFINE_boolean("gpu_allow_growth", False,
                        """Allow GPU memory allocation to grow
                        dynamically.""")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        """Log the op placement to devices""")


FLAGS = tf.flags.FLAGS

# params
HParams = namedtuple(
  "HParams",
  [ "cell",
    "batch_size",
    "layers",
    "attention",
    "source_vocab_path",
    "target_vocab_path",
    "rl_training",  
    "enc_embedding_dim",
    "dec_embedding_dim",
    "hidden_size",
    "attn_size",
    "eval_batch_size",
    "learning_rate",
    "max_source_len",
    "max_target_len",
    "optimizer",
    "optimizer_clip_gradients"])

# create params
def create_hparams():
    if FLAGS.cell_model == "LSTM":
        cell_model = tf.contrib.rnn.LSTMCell
    else:
        cell_model = tf.contrib.rnn.GRUCell
        
    return HParams(
        cell=cell_model,
        batch_size=FLAGS.batch_size,
        rl_training=FLAGS.rl_training,
        source_vocab_path=FLAGS.source_vocab_path,
        target_vocab_path=FLAGS.target_vocab_path,
        layers=FLAGS.num_layers,
        eval_batch_size=FLAGS.eval_batch_size,
        attention=FLAGS.attention,
        optimizer=FLAGS.optimizer,
        optimizer_clip_gradients=FLAGS.optimizer_clip_gradients, #10.0,
        learning_rate=FLAGS.learning_rate,
        enc_embedding_dim=FLAGS.embedding_dim,
        dec_embedding_dim=FLAGS.embedding_dim,
        hidden_size=FLAGS.hidden_size,
        attn_size=FLAGS.hidden_size,
        max_source_len=FLAGS.max_source_len,
        max_target_len=FLAGS.max_target_len)

def create_experiment(output_dir):
    config = run_config.RunConfig(
        tf_random_seed=FLAGS.tf_random_seed,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config.tf_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth
    config.tf_config.log_device_placement = FLAGS.log_device_placement

    #params
    hparams = create_hparams()
    #print(hparams)

    #Create train input function
    train_input_fn = create_input_fn(source_file_list=FLAGS.source_files,
                                     target_file_list=FLAGS.target_files,
                                     batch_size=FLAGS.batch_size,
                                     scope="train_input_fn")
    
    eval_input_fn = create_input_fn(source_file_list=FLAGS.dev_source_files,
                                    target_file_list=FLAGS.dev_target_files,
                                    batch_size=FLAGS.batch_size,
                                    scope="dev_input_fn")
    
    #vocab info 
    source_vocab_info = vocab.get_vocab_info(hparams.source_vocab_path)
    target_vocab_info = vocab.get_vocab_info(hparams.target_vocab_path)

    #seq2seq model_fn
    seq2seq = Seq2Seq(target_vocab_info.special_vocab.SEQUENCE_START,
                      target_vocab_info.special_vocab.SEQUENCE_END,
                      hparams)

    #estimator
    estimator = tf.contrib.learn.Estimator(
        model_fn=seq2seq.create_model_fn(),
        model_dir=output_dir,
        config=config,
        params=hparams)
        
    #Create metrics
    eval_metrics = {
        'bleu' : tf.contrib.learn.MetricSpec(
            metric_fn=bleu_fn,
            prediction_key="predictions", 
            label_key="target_tokens")
    }
    
    #train_hooks
    train_hooks = []
    
    #experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        min_eval_frequency=FLAGS.eval_every_n_steps,
        train_steps=FLAGS.train_steps,
        eval_steps=None,
        eval_metrics=eval_metrics,
        train_monitors=train_hooks)
   
    return experiment

def main(_argv):
    if FLAGS.save_checkpoints_secs is None \
        and FLAGS.save_checkpoints_steps is None:
        FLAGS.save_checkpoints_secs = 600
        tf.logging.info("Setting save_checkpoints_secs to %d", FLAGS.save_checkpoints_secs)

    if not FLAGS.source_vocab_path or not FLAGS.target_vocab_path:
        raise ValueError("You must specify source_vocab_path and target_vocab_path")
        
    if not FLAGS.output_dir:
        FLAGS.output_dir = tempfile.mkdtemp()
    
    if not FLAGS.source_files or not FLAGS.target_files:
        raise ValueError("You must specify source_path and target_path")

    FLAGS.source_files = FLAGS.source_files.strip().split(',')
    print(FLAGS.source_files)

    FLAGS.target_files = FLAGS.target_files.strip().split(',')

    if not FLAGS.dev_source_files or not FLAGS.dev_target_files:
        raise ValueError("You must specify dev_*_path")

    learn_runner.run(
      experiment_fn=create_experiment,
      output_dir=FLAGS.output_dir,
      schedule=FLAGS.schedule)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run() 
