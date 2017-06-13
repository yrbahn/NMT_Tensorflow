import numpy as np
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

# SEQUENCE_START 내@@ 림 B@@ 음@@ . SEQUENCE_END
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
 
def calculate_advantage(target_tokens_list, 
                        sampling_tokens_list, 
                        greedy_tokens_list):
    advantage = []
    for i, target_tokens in enumerate(target_tokens_list):
        sampling_tokens = decode_and_slice_tokens(sampling_tokens_list[i].tolist())    
        greedy_tokens   = decode_and_slice_tokens(greedy_tokens_list[i].tolist())
        target_tokens   = decode_and_slice_tokens(target_tokens.tolist())
        
        baseline_blue_score = nltk.translate.bleu_score.sentence_bleu([target_tokens], greedy_tokens)
        blue_score = nltk.translate.bleu_score.sentence_bleu([target_tokens], sampling_tokens)
        advantage.append(baseline_blue_score - blue_score)
    return np.array(advantage, dtype=np.float32)

            
        
        
        
