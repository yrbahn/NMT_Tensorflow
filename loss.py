import tensorflow as tf
from utils import tf_print

def rl_loss(logits, targets, advantage, weights, name=None):

        if len(logits.get_shape()) != 3:
            raise ValueError("Logits must be a "
                             "[batch_size x sequence_length x logits] tensor")
        
        if len(targets.get_shape()) != 2:
            raise ValueError("Targets must be a [batch_size x sequence_length] "
                         "tensor")
        
        if len(weights.get_shape()) != 2:
            raise ValueError("Weights must be a [batch_size x sequence_length] "
                             "tensor")
        with tf.name_scope(name, "sequence_loss", [logits, targets, weights]):
            num_classes = tf.shape(logits)[2]
            probs_flat = tf.nn.softmax(tf.reshape(logits, [-1, num_classes]))
            targets = tf.reshape(targets, [-1])
            
            probs = tf.reduce_sum(probs_flat * tf.one_hot(targets, depth=num_classes), 1)
            logprobs = tf.log(probs)
            
            batch_size = tf.shape(logits)[0]
            sequence_length = tf.shape(logits)[1]
            logprobs = tf.reshape(logprobs, [batch_size, sequence_length])

            #J = -logprobs * advantage[:, None] 
            advantage = tf_print(advantage, "advantage:")
            losses = -tf.reduce_sum(logprobs*weights, 1) * advantage
            #losses = tf_print(losses, "losses:")

            #regularize with entropy            
            probs = tf.reshape(probs, [batch_size, sequence_length])
            entropy = -probs * logprobs 
            entropy = tf.reduce_sum(entropy*weights)
            #entropy /= total_size
            
            losses = losses - 0.01*entropy

            loss = tf.reduce_mean(losses)

            return loss
"""
def rl_loss(logits, targets, advantage, weights, name=None):

        if len(logits.get_shape()) != 3:
            raise ValueError("Logits must be a "
                             "[batch_size x sequence_length x logits] tensor")
        
        if len(targets.get_shape()) != 2:
            raise ValueError("Targets must be a [batch_size x sequence_length] "
                         "tensor")
        
        if len(weights.get_shape()) != 2:
            raise ValueError("Weights must be a [batch_size x sequence_length] "
                             "tensor")
        with tf.name_scope(name, "sequence_loss", [logits, targets, weights]):
            num_classes = tf.shape(logits)[2]
            #num_classes = tf.Print(num_classes, [num_classes], message='n_classes')
                     
            probs_flat = tf.nn.softmax(tf.reshape(logits, [-1, num_classes]))
            targets = tf.reshape(targets, [-1])
            
            probs = tf.reduce_sum(probs_flat * tf.one_hot(targets, depth=num_classes), 1)
            #probs = tf.Print(probs, [probs], message="probs:")
            logprobs = tf.log(probs)
             
            crossent = -logprobs * tf.reshape(weights, [-1])
            #crossent = tf.Print(crossent, [crossent], message='crossent:')
 
            batch_size = tf.shape(logits)[0]
            sequence_length = tf.shape(logits)[1]
            crossent = tf.reshape(crossent, [batch_size, sequence_length])

            J = tf.reduce_sum(crossent*weights, 1) * tf.reshape(advantage, [-1]) 
            #J = tf.Print(J, [J], message='J:')

            #entropy = -probs * logprobs 
            #entropy = tf.reshape(entropy, [batch_size, sequence_length])
            #entropy = tf.reduce_sum(entropy*weights, 1)
            #entropy = tf.Print(entropy, [entropy], message='entropy:') 
            #loss = tf.reduce_sum(J - 0.01*entropy)
            loss = tf.reduce_mean(J)

            return loss
"""
