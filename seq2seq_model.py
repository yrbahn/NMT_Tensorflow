import tensorflow as tf
import vocab

from loss import rl_loss
from sample_embedding_helper import SampleEmbeddingHelper
from utils import get_available_gpus, calculate_advantage

from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# sequence to sequence model
class Seq2Seq(object):
    # __init__
    def __init__(self, target_bos_id, target_eos_id, params):
        self.params = params
        #self.mode = mode
        self.available_gpus = get_available_gpus()
        self.current_gpu_index = 0
        self.total_gpu_num = len(self.available_gpus)
        self.target_bos_id = target_bos_id
        self.target_eos_id = target_eos_id
 
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
        self.source_tokens     = tf.placeholder_with_default(source_tokens, shape=[None,None], name='source_tokens')
        self.source_tokens_len = tf.placeholder_with_default(source_tokens_len, shape=[None], name='source_tokens_len')

        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.target_tokens     = tf.placeholder_with_default(target_tokens, shape=[None, None], name='target_tokens')
            self.target_tokens_len = tf.placeholder_with_default(target_tokens_len, shape=[None], name='target_tokens_len')
        else:
            self.target_tokens = None
   
        #rl enable var
        print(self.params.rl_training)
        self.rl_training   = tf.placeholder_with_default(self.params.rl_training, shape=None, name="rl_training") 
        
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
        self.input_ids = source_vocab_to_id.lookup(self.source_tokens[:,:self.params.max_source_len])
        self.inputs_len = tf.minimum(self.source_tokens_len, self.params.max_source_len)

        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.target_ids = target_vocab_to_id.lookup(self.target_tokens[:,:self.params.max_target_len])
            self.targets_len = tf.minimum(self.target_tokens_len, self.params.max_target_len)
               
            # set max_target_len
            self.max_target_len = tf.reduce_max(self.targets_len-1, name='max_target_len')
        else:
            # set max_target_len
            self.max_target_len = self.params.max_target_len
            
    # add a encoder
    def _add_encoder(self):
        with tf.variable_scope('Encoder') as scope:
            
            self.batch_size = tf.shape(self.input_ids)[0]

            enc_W_emb = tf.get_variable('en_embedding', 
                                        initializer=tf.random_uniform([self.source_vocab_size, self.params.enc_embedding_dim]),
                                        dtype=tf.float32)
            
            enc_emb_inputs = tf.nn.embedding_lookup(
                enc_W_emb, self.input_ids, name='emb_inputs')

            # bidirectional rnn
            if self.params.layers == 1:
                enc_cell = tf.contrib.rnn.DeviceWrapper(
                    self.params.cell(self.params.hidden_size), 
                    device=self._get_next_gpu())
                
                self.enc_outputs, self.enc_last_state = tf.nn.dynamic_rnn(
                    cell=enc_cell,
                    inputs=enc_emb_inputs,
                    sequence_length=self.inputs_len,
                    time_major=False,
                    dtype=tf.float32)                
            else:
                enc_cell_fw = self.params.cell(self.params.hidden_size)
                enc_cell_fw = tf.contrib.rnn.DeviceWrapper(enc_cell_fw, device=self._get_next_gpu())

                gpu2 = self._get_next_gpu()

                enc_cell_bw = self.params.cell(self.params.hidden_size)
                enc_cell_bw = tf.contrib.rnn.DeviceWrapper(enc_cell_bw, device=gpu2)

                enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, 
                                                                           enc_cell_bw, 
                                                                           enc_emb_inputs,
                                                                           self.inputs_len,
                                                                           dtype=tf.float32)
                enc_outputs = tf.concat(enc_outputs,2)

                enc_cell = tf.contrib.rnn.DeviceWrapper(
                        self.params.cell(num_units=self.params.hidden_size),
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
                
                self.enc_last_state = (enc_state[0],) + self.enc_last_state
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
                        normalize=False,
                        name='LuongAttention')

                self.dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
                        cell=self.dec_cell,
                        attention_mechanism=attn_mech,
                        attention_size=self.params.attn_size,
                        # attention_history=False (in ver 1.2)
                        name='Attention_Wrapper')

                # last_enc_state wrapper
                self.initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(
                        cell_state=self.enc_last_state,
                        attention=_zero_state_tensors(self.params.attn_size, self.batch_size, tf.float32))
            else:
                self.initial_state = self.enc_last_state

            self.dec_W_emb = tf.get_variable('de_embedding', 
                                             initializer=tf.random_uniform([self.target_vocab_size,
                                                                                        self.params.dec_embedding_dim]),
                                             dtype=tf.float32)
                

            self.output_layer = Dense(self.target_vocab_size, name='output_projection')
            if mode == tf.contrib.learn.ModeKeys.TRAIN: 
                # training layer
                self.predictions, self.greedy_predictions, self.loss = tf.cond(self.rl_training, 
                                                                               lambda : self._add_rl_training_layer(scope), 
                                                                               lambda : self._add_training_layer(scope))
                #self.predictions, self.greedy_predictions, self.loss = self._add_training_layer(scope)
            else: # inference layer
                self.predictions, self.loss = self._add_inference_layer(mode) 
                
    # inference layer
    def _add_inference_layer(self, mode):
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

            inference_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                inference_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_target_len)

            predictions = inference_outputs.sample_id
        else: # tensorflow 1.2 only(no test)
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
            self.max_target_len = tf.shape(predictions)[1]
            masks = tf.sign(tf.to_float(predictions))
            #masks = tf.sequence_mask(self.targets_len-1, self.max_target_len, dtype=tf.float32, name='masks')
            loss = tf.contrib.seq2seq.sequence_loss(logits=inference_outputs.rnn_output,
                                                         targets=self.target_ids[:,1:self.max_target_len+1],
                                                         weights=masks, name='batch_loss')
     
        return predictions, loss
        
    
    # training layer
    def _add_training_layer(self, scope):
        # training_layer        
        dec_inputs = self.target_ids[:,:-1]
        
        dec_emb_inputs = tf.nn.embedding_lookup(
            self.dec_W_emb, dec_inputs, name='emb_inputs')

        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=dec_emb_inputs,
            sequence_length=self.targets_len-1,
            time_major=False,
            name='training_helper')

        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.dec_cell,
            helper=training_helper,
            initial_state=self.initial_state,
            output_layer=self.output_layer)

        #self.max_target_len = tf.reduce_max(self.targets_len-1, name='max_target_len')
        
        train_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=self.max_target_len)
            
        # predictions
        predictions = train_outputs.sample_id

        masks = tf.sequence_mask(self.targets_len-1, self.max_target_len, dtype=tf.float32, name='masks')

        # loss
        loss = tf.contrib.seq2seq.sequence_loss(logits=train_outputs.rnn_output, 
                                                     targets=self.target_ids[:,1:],
                                                     weights=masks, name='batch_loss')        
        return predictions, predictions, loss
        
    def _add_rl_training_layer(self, scope):
        # RL training_layer
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

        greedy_dec_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
            greedy_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=self.max_target_len)

        # greedy predictions
        greedy_predictions = greedy_dec_outputs.sample_id
        
        scope.reuse_variables()

        #sampling decoding
        training_sampling_helper = SampleEmbeddingHelper(
            embedding=self.dec_W_emb,
            start_tokens=start_tokens,
            end_token=self.target_eos_id)

        training_sampling_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.dec_cell,
            helper=training_sampling_helper,
            initial_state=self.initial_state,
            output_layer=self.output_layer)

        train_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
            training_sampling_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=self.max_target_len)

        #seq_len = tf.shape(train_outputs.rnn_output)[1]

        # predictions
        predictions = train_outputs.sample_id

        #calculate advantage
        prediction_tokens = self.target_id_to_vocab.lookup(tf.to_int64(predictions))
        greedy_prediction_tokens = self.target_id_to_vocab.lookup(tf.to_int64(greedy_predictions))
        target_tokens = self.target_tokens[:,1:]
        advantage = tf.py_func(calculate_advantage, 
                               [target_tokens, prediction_tokens, greedy_prediction_tokens], 
                               Tout=tf.float32)
        
        advantage = tf.reshape(advantage, [self.batch_size])

        # mask
        masks = tf.sign(tf.to_float(predictions))

        # loss
        loss = rl_loss(logits=train_outputs.rnn_output, targets=predictions,
                            advantage=advantage, weights=masks, name='rl_loss')
        return predictions, greedy_predictions, loss

    # optimizing layer
    def _add_optimizer(self):
        with tf.variable_scope('Optimizer') as scope:
            def _clip_gradients(grads_and_vars):
                """Clips gradients by global norm."""
                gradients, variables = zip(*grads_and_vars)
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, self.params.optimizer_clip_gradients)
                return list(zip(clipped_gradients, variables))

            self.train_op = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=self.params.learning_rate, 
                clip_gradients=_clip_gradients,
                optimizer=self.params.optimizer)

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
                self._add_optimizer()
                return model_fn_lib.ModelFnOps(
                    mode=mode,
                    predictions={'predictions': self.predictions,
                                 #'sampling_sent': self.target_id_to_vocab.lookup(self.predictions),
                                 'greedy_predictions': self.greedy_predictions,
                                 #'greedy_sent': self.target_id_to_vocab.lookup(tf.to_int64(self.greedy_predictions))
                                },
                                 #'J': j,
                                 #'crossent': crossent},
                    loss=self.loss,
                    train_op=self.train_op)
            else:
                return model_fn_lib.ModelFnOps(
                    mode=mode,
                    predictions={'predictions': 
                                 self.target_id_to_vocab.lookup(tf.to_int64(self.predictions))},
                    loss=self.loss
                )
        return model_fn
          
