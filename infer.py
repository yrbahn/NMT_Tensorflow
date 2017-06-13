import tensorflow as tf

tf.flags.DEFINE_string("model_dir", None, "directory to load model from")
tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")
tf.flags.DEFINE_string("checkpoint_path", None,
                       """Full path to the checkpoint to be loaded. If None,
                       the latest checkpoint in the model dir is used.""")

FLAGS = tf.flags.FLAGS

def main(_argv):
    
    saver = tf.train.Saver()
    checkpoint_path = FLAGS.checkpoint_path
    if not checkpoint_path:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    
    