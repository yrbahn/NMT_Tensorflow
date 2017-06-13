import tensorflow as tf

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
            
            crossent = -logprobs * tf.reshape(weights, [-1])
            
            batch_size = tf.shape(logits)[0]
            sequence_length = tf.shape(logits)[1]
            crossent = tf.reshape(crossent, [batch_size, sequence_length])

            J = crossent * advantage[:, None] 

            loss = tf.reduce_sum(J)
            total_size = tf.reduce_sum(weights)
            loss /= total_size

            #regularize with negative entropy
            entropy = -probs * logprobs 
            entropy = tf.reduce_sum(entropy)
            total_size = tf.reduce_sum(weights)
            entropy /= total_size

            loss -=  0.01*entropy
        
            return loss