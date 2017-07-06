import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
import nltk

def to_sent_string(word_list, length=None, end_token=b"SEQUENCE_END"):
    if length:
        return ' '.join(np.char.decode(word_list[:length], "utf-8"))
    else:
        try:
            i = word_list.index(end_token)
            return ' '.join(np.char.decode(word_list[:i], "utf-8"))
        except :
            return ' '.join(np.char.decode(word_list, "utf-8"))
            
            
def decode_and_slice_tokens(word_list, length=None, end_token=b"SEQUENCE_END"):
    if length:
        return np.char.decode(word_list[:length], "utf-8")
    else:
        try:
            i = word_list.index(end_token)
            return np.char.decode(word_list[:i], "utf-8")
        except :
            return np.char.decode(word_list, "utf-8")

# get available gpus
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def slice_text(text,
               eos_token="SEQUENCE_END",
               sos_token="SEQUENCE_START"):
    """Slices text from SEQUENCE_START to SEQUENCE_END, not including
    these special tokens.
    """
    eos_index = text.find(eos_token)
    text = text[:eos_index] if eos_index > -1 else text
    sos_index = text.find(sos_token)
    text = text[sos_index+len(sos_token):] if sos_index > -1 else text

    return text


def compute_bleu(raw_sentences, sentence_masks, raw_targets, target_masks):

    #print("raw_sentences:", raw_sentences)
    sentences = []
    for i, sent in enumerate(raw_sentences):
        sentences.append(sent[:int(sum(sentence_masks[i]))])
     
    targets = []
    for i, target in enumerate(raw_targets):
        targets.append(target[:int(sum(target_masks[i]))])
                         
    blue_scores = []
    cc = nltk.translate.bleu_score.SmoothingFunction()
    for sent, target in zip(sentences, targets):
        score = int(nltk.translate.bleu_score.sentence_bleu([target], sent, smoothing_function=cc.method4)*10)
        
        blue_scores.append(score)
    return np.array(blue_scores, np.float32)
 
def estimate_advantage(targets,
                        target_masks,
                        sampling_sentences,
                        sampling_sentence_masks,
                        greedy_sentences,
                        greedy_sentence_masks):
    
    baseline_scores = compute_bleu(greedy_sentences, 
                                   greedy_sentence_masks,
                                   targets,
                                   target_masks)
                       
    reward_scores = compute_bleu(sampling_sentences,
                                 sampling_sentence_masks,
                                 targets,
                                 target_masks)
                       
    advantage = reward_scores - baseline_scores
    return advantage

def get_mask_by_eos(is_eos):
    is_right_after_eos = tf.concat([tf.zeros_like(is_eos[:,:1]),is_eos[:,:-1]],-1)
    is_after_eos = tf.equal(tf.cumsum(tf.to_float(is_right_after_eos),axis=-1), 0)
    return tf.to_float(is_after_eos)

def tf_print(tensor, message=''):
    def print_tensor(x):
        print(message, x)
        return x
    
    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    return res
