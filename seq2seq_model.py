import tensorflow as tf
import vocab

from loss import rl_loss
#from sample_embedding_helper import SampleEmbeddingHelper
from utils import get_available_gpus, get_mask_by_eos, compute_bleu, tf_print

from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# sequence to sequence model
class Seq2Seq(object):
    # __init__
    def __init__(self, target_bos_id, target_eos_id, params):
        self.params = params
        self.available_gpus = get_available_gpus()
        self.current_gpu_index = 0
        self.total_gpu_num = len(self.available_gpus)
        self.target_bos_id = target_bos_id
        self.target_eos_id = target_eos_id
 
        print("learning_rate:", self.params.learning_rate)
    
    # get the next gpu
    def _get_next_gpu(self):
        if self.total_gpu_num == 0:
            return 'cpu:0'
        else:
            self.current_gpu_index %= self.total_gpu_num
            current_gpu = self.available_gpus[self.current_gpu_index]
            self.current_gpu_index += 1
            return current_gpu
        
    # add placeholder variables
    def _add_placeholders(self, 
                          source_tokens=None,
                          source_tokens_len=None,
                          target_tokens=None,
                          target_tokens_len=None, 
                          mode=tf.contrib.learn.ModeKeys.TRAIN):
        
        #assign placeholder variables
        self.source_tokens     = tf.placeholder_with_default(source_tokens, 
                                                             shape=[None,None], 
                                                             name='source_tokens')
        
        self.source_tokens_len = tf.placeholder_with_default(source_tokens_len, 
                                                             shape=[None], 
                                                             name='source_tokens_len')

        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.target_tokens     = tf.placeholder_with_default(target_tokens, 
                                                                 shape=[None, None], 
                                                                 name='target_tokens')
            self.target_tokens_len = tf.placeholder_with_default(target_tokens_len, 
                                                                 shape=[None], 
                                                                 name='target_tokens_len')
        else:
            self.target_tokens = None
   
        # rl training
        self.rl_training = self.params.rl_training
        
    # set variables for inputs and targets
    def _set_variables(self, mode):
        source_vocab_to_id, source_id_to_vocab, source_word_to_count, source_vocab_size = \
            vocab.create_vocabulary_lookup_table(self.params.source_vocab_path) #"./data/vocab_en.txt")
        target_vocab_to_id, self.target_id_to_vocab, target_word_to_count, target_vocab_size = \
            vocab.create_vocabulary_lookup_table(self.params.target_vocab_path) # "./data/vocab_kr.txt")

        # set vocab size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        # set inputs and targets
        self.source_tokens = self.source_tokens[:,:self.params.max_source_len]
        self.input_ids     = source_vocab_to_id.lookup(self.source_tokens)
        self.inputs_len    = tf.minimum(self.source_tokens_len, self.params.max_source_len)
        
        self.max_target_len = self.params.max_target_len
        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.target_tokens = self.target_tokens[:,:self.max_target_len]
            self.target_ids    = target_vocab_to_id.lookup(self.target_tokens)
            self.targets_len   = tf.minimum(self.target_tokens_len, self.params.max_target_len)
            
            if mode == tf.contrib.learn.ModeKeys.EVAL or not self.params.rl_training:
                self.max_target_len = tf.reduce_max(self.targets_len, name='max_target_len')
        
            
    # add a encoder
    def _add_encoder(self):
        with tf.variable_scope('Encoder') as scope:
            
            self.batch_size = tf.shape(self.input_ids)[0]

            enc_W_emb = tf.get_variable('en_embedding', 
                                        initializer=tf.random_uniform([self.source_vocab_size, 
                                                                       self.params.enc_embedding_dim]),
                                        dtype=tf.float32)
            
            enc_emb_inputs = tf.nn.embedding_lookup(
                enc_W_emb, self.input_ids, name='emb_inputs')

            # bidirectional rnn
            if self.params.layers == 1:
                enc_cell = tf.contrib.rnn.DeviceWrapper(
                    tf.contrib.rnn.DropoutWrapper(
                        self.params.cell(self.params.hidden_size),
                        self.params.output_keep_prob), device=self._get_next_gpu())
                
                self.enc_outputs, self.enc_last_state = tf.nn.dynamic_rnn(
                    cell=enc_cell,
                    inputs=enc_emb_inputs,
                    sequence_length=self.inputs_len,
                    time_major=False,
                    dtype=tf.float32)                
            else:
                enc_cell_fw = tf.contrib.rnn.DropoutWrapper(
                    self.params.cell(self.params.hidden_size), 
                    output_keep_prob=self.params.output_keep_prob)

                enc_cell_fw = tf.contrib.rnn.DeviceWrapper(enc_cell_fw, device=self._get_next_gpu())

                gpu2 = self._get_next_gpu()

                enc_cell_bw = tf.contrib.rnn.DropoutWrapper(
                    self.params.cell(self.params.hidden_size), 
                    output_keep_prob=self.params.output_keep_prob)

                enc_cell_bw = tf.contrib.rnn.DeviceWrapper(enc_cell_bw, device=gpu2)

                enc_outputs, enc_states = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, 
                                                                         enc_cell_bw, 
                                                                         enc_emb_inputs,
                                                                         self.inputs_len,
                                                                         dtype=tf.float32)
                #merge outputs, states
                enc_outputs = tf.concat(enc_outputs,2)
                enc_state   = tf.add(enc_states[0], enc_states[1])
                
                enc_cell = tf.contrib.rnn.DeviceWrapper(
                    tf.contrib.rnn.DropoutWrapper(
                        self.params.cell(num_units=self.params.hidden_size),
                        output_keep_prob=self.params.output_keep_prob),
                    device=gpu2)

                # multi rnn
                if self.params.layers > 2:
                    enc_cell = [enc_cell]
                    for _ in range(self.params.layers-2):
                        enc_cell.append(tf.contrib.rnn.DeviceWrapper(
                            tf.contrib.rnn.ResidualWrapper(
                                self.params.cell(num_units=self.params.hidden_size)),
                            device=self._get_next_gpu()))

                    enc_cell = tf.contrib.rnn.MultiRNNCell(enc_cell)
                    
                self.enc_outputs, self.enc_last_state = tf.nn.dynamic_rnn(
                    cell=enc_cell,
                    inputs=enc_outputs,
                    sequence_length=self.inputs_len,
                    time_major=False,
                    dtype=tf.float32)
                
                if type(self.enc_last_state) is tuple:
                    self.enc_last_state = (enc_state,) + self.enc_last_state
                else:
                    self.enc_last_state = (enc_state, self.enc_last_state)
    
    def _clip_gradients(self, grads_and_vars):
        """Clips gradients by global norm."""
        gradients, variables = zip(*grads_and_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.params.optimizer_clip_gradients)
        return list(zip(clipped_gradients, variables))
    
    # add an encoder
    def _add_decoder(self, mode):
        
        with tf.variable_scope('Decoder') as scope:
            cells = []
            if self.params.layers > 1:
                for i in range(self.params.layers):
                    if i == 0:
                        cells.append(tf.contrib.rnn.DeviceWrapper(
                            self.params.cell(self.params.hidden_size), 
                            device=self._get_next_gpu()))
                    else:
                        cells.append(tf.contrib.rnn.DeviceWrapper(
                            tf.contrib.rnn.ResidualWrapper(
                                self.params.cell(num_units=self.params.hidden_size)),
                            device=self._get_next_gpu()))

                self.dec_cell = tf.contrib.rnn.MultiRNNCell(cells)
            else:
                self.dec_cell = tf.contrib.rnn.DeviceWrapper(
                                    self.params.cell(self.params.hidden_size), 
                                    device=self._get_next_gpu())
            
            if self.params.attention:
                attn_mech = tf.contrib.seq2seq.LuongAttention(
                        num_units=self.params.attn_size,
                        memory=self.enc_outputs,
                        memory_sequence_length=self.inputs_len,
                        name='LuongAttention')

                self.dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                        cell=self.dec_cell,
                        attention_mechanism=attn_mech,
                        attention_layer_size=self.params.attn_size,
                        name='Attention_Wrapper')
                
                attention_zero = self.dec_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                
                # last_enc_state wrapper
                self.initial_state = attention_zero.clone(cell_state=self.enc_last_state)
            else:
                self.initial_state = self.enc_last_state

            self.dec_W_emb = tf.get_variable('de_embedding', 
                                             initializer=tf.random_uniform([self.target_vocab_size,
                                                                            self.params.dec_embedding_dim]),
                                             dtype=tf.float32)
                
            self.output_layer = Dense(self.target_vocab_size, name='output_projection')
            if mode == tf.contrib.learn.ModeKeys.TRAIN: 
                # training layer
                if self.rl_training:
                    self.predictions, self.greedy_predictions, self.loss = self._add_rl_training_layer()
                else:
                    self.predictions, self.loss = self._add_training_layer()
               
                
                #optimizer
                self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                                global_step=tf.contrib.framework.get_global_step(),
                                                                learning_rate=self.params.learning_rate, 
                                                                clip_gradients=self._clip_gradients,
                                                                optimizer=self.params.optimizer)
            else: # inference layer
                self.predictions, self.loss = self._add_inference_layer(mode) 
                
    # inference layer
    def _add_inference_layer(self, mode):
        with tf.name_scope("inference_layer"):
            # inference layer        
            if not self.params.beam_search:
                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.dec_W_emb,
                    start_tokens=tf.fill([self.batch_size], self.target_bos_id),
                    end_token=self.target_eos_id) 

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.dec_cell,
                    helper=inference_helper,
                    initial_state=self.initial_state,
                    output_layer=self.output_layer)

                inference_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.max_target_len)

                predictions = inference_outputs.sample_id
            else: # TODO: beam_decoding
                bs_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=cell,
                    embedding=self.dec_W_emb,
                    start_tokens=tf.fill([self.batch_size], self.target_bos_id),
                    end_token=self.target_eos_id,
                    initial_state=self.initial_state,
                    beam_width=self.params.beam_width,
                    output_layer=self.output_layer,
                    length_penalty_weight=self.params.length_penalty_weight)

                inference_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    bs_decoder,
                    output_time_major=False,
                    maximum_iterations=self.max_target_len)

                #max_sequence_length = 
                predictions = inference_outputs.predicted_ids[:,:,0]

            loss = None
            if mode == tf.contrib.learn.ModeKeys.EVAL:# EVAL
                max_prediction_len = tf.shape(predictions)[1]
                masks = get_mask_by_eos(tf.equal(predictions, self.target_eos_id))
                loss = tf.contrib.seq2seq.sequence_loss(logits=inference_outputs.rnn_output,
                                                             targets=self.target_ids[:,:max_prediction_len],
                                                             weights=masks, name='batch_loss')

            return predictions, loss

    
    # training layer
    def _add_training_layer(self):
        with tf.name_scope("training_layer"):
            # training_layer        
            dec_inputs = tf.concat([tf.zeros_like(self.target_ids[:,:1])+self.target_bos_id,
                                    self.target_ids[:,:-1]],axis=1)

            #dec_inputs = tf_print(dec_inputs, "dec_inputs:")
            #self.targets_len = tf_print(self.targets_len, "target_len:")

            dec_emb_inputs = tf.nn.embedding_lookup(
                self.dec_W_emb, dec_inputs, name='emb_inputs')

            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=dec_emb_inputs,
                sequence_length=self.targets_len,
                time_major=False,
                name='training_helper')

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.dec_cell,
                helper=training_helper,
                initial_state=self.initial_state,
                output_layer=self.output_layer)

            train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                training_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_target_len)

            # predictions
            predictions = train_outputs.sample_id

            masks = tf.sequence_mask(self.targets_len, self.max_target_len, dtype=tf.float32, name='masks')

            # loss
            loss = tf.contrib.seq2seq.sequence_loss(logits=train_outputs.rnn_output, 
                                                    targets=self.target_ids,
                                                    weights=masks, 
                                                    name='batch_loss')        
            return predictions, loss
    
    # rl training
    def _add_rl_training_layer(self):
        with tf.name_scope("rl_training_layer") as scope:
            #targets, masks
            #self.target_tokens = tf_print(self.target_tokens, "target_sent:")
            target_masks = get_mask_by_eos(tf.equal(self.target_ids, self.target_eos_id))

            #start_tokens
            sequence_start = [self.target_bos_id]
            start_tokens = tf.tile(sequence_start, [self.batch_size], name='start_tokens')

            #greedy decoding
            greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.dec_W_emb,
                start_tokens=start_tokens,
                end_token=self.target_eos_id) 

            greedy_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.dec_cell,
                helper=greedy_helper,
                initial_state=self.initial_state,
                output_layer=self.output_layer)

            greedy_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                greedy_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_target_len)

            # greedy predictions
            greedy_predictions = greedy_dec_outputs.sample_id

            greedy_prediction_masks = get_mask_by_eos(tf.equal(greedy_predictions, self.target_eos_id))

            greedy_sentences = self.target_id_to_vocab.lookup(tf.to_int64(greedy_predictions))
            #greedy_sentences = tf_print(greedy_sentences, "greedy_sent:")

            #estimsate baseline
            baseline = tf.py_func(compute_bleu, 
                                  [greedy_sentences, 
                                   greedy_prediction_masks, 
                                   self.target_tokens, 
                                   target_masks],
                                  Tout=tf.float32)

            baseline = tf.stop_gradient(tf.reshape(baseline, [self.batch_size]))

            #scope.reuse_variables()

            #sampling decoding
            training_sampling_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                embedding=self.dec_W_emb,
                start_tokens=start_tokens,
                end_token=self.target_eos_id)

            training_sampling_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.dec_cell,
                helper=training_sampling_helper,
                initial_state=self.initial_state,
                output_layer=self.output_layer)

            train_sampling_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                training_sampling_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_target_len)

            # sampled predictions
            predictions = train_sampling_outputs.sample_id

            sampling_masks = get_mask_by_eos(tf.equal(predictions, self.target_eos_id))

            sampling_sentences = self.target_id_to_vocab.lookup(tf.to_int64(predictions))

            #sampling_sentences = tf_print(sampling_sentences, "sampling_sent:")

            rewards = tf.py_func(compute_bleu, 
                                 [sampling_sentences, 
                                  sampling_masks, 
                                  self.target_tokens, 
                                  target_masks],
                                 Tout=tf.float32)

            rewards = tf.stop_gradient(tf.reshape(rewards, [self.batch_size]))

            with tf.control_dependencies([baseline, rewards]):
                #estimate advantage
                advantage = rewards - baseline 
                #advantage = tf_print(advantage, "advantage:")

                # loss
                loss = rl_loss(logits=train_sampling_outputs.rnn_output, 
                               targets=predictions,
                               advantage=advantage, 
                               weights=sampling_masks, 
                               name='rl_loss')

                return predictions, greedy_predictions, loss

   
    # create model_fn for estimator
    def create_model_fn(self):        
        
        # model_fn
        def model_fn(features, labels, params, mode): 
            #get inputs, targets from features, labels
            source_tokens = tf.placeholder_with_default(features["source_tokens"], 
                                                  shape=[None, None],
                                                  name='sentence_inputs')
            source_tokens_length = tf.placeholder_with_default(features["source_len"], 
                                                        shape=[None],
                                                        name='source_tokens_length')

            if labels != None:
                target_tokens = tf.placeholder_with_default(labels["target_tokens"],
                                                  shape=[None, None],
                                                  name='targets')

                target_tokens_length = tf.placeholder_with_default(labels["target_len"],
                                                             shape=[None],
                                                             name='targets_token_length')
            else:
                target_tokens = None
                target_tokens_length = None

            # add placeholders
            self._add_placeholders(source_tokens, source_tokens_length,
                                   target_tokens, target_tokens_length,
                                   mode)

            # set input_ids, target_ids, max_target_len
            self._set_variables(mode)
           
            # add an encoder to computation graph
            self._add_encoder()
            
            # add a decoder
            self._add_decoder(mode)
                                
            if mode == tf.contrib.learn.ModeKeys.TRAIN: # train
                return model_fn_lib.ModelFnOps(
                    mode=mode,
                    predictions={'predictions': self.predictions },
                    loss=self.loss,
                    train_op=self.train_op)   
            else:
                return model_fn_lib.ModelFnOps(
                    mode=mode,
                    predictions={'predictions': 
                                 self.target_id_to_vocab.lookup(tf.to_int64(self.predictions)),
                                 'source_tokens': self.source_tokens},
                    loss=self.loss
                )
        return model_fn
          
