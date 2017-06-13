import tensorflow as tf

class SampleEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
    def __init__(self, embedding, start_tokens, end_token,seed=None):
        super(SampleEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token)
        self._seed = seed

    def sample(self, time, outputs, state, name=None):
        """sample for SamplingEmbeddingHelper"""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, tf.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                          type(outputs))
        sample_id_sampler = tf.contrib.distributions.Categorical(logits=outputs)
        sample_ids = sample_id_sampler.sample(seed=self._seed)
        return sample_ids
