import numpy as np
import six

import tensorflow as tf
from tensorflow.contrib import metrics
from tensorflow.contrib.learn import MetricSpec

import bleu
import utils

def accumulate_strings(values, name="strings"):
    tf.assert_type(values, tf.string)
    strings = tf.Variable(
        name=name,
        initial_value=[],
        dtype=tf.string,
        trainable=False,
        collections=[],
        validate_shape=True)
    value_tensor = tf.identity(strings)
    update_op = tf.assign(
        ref=strings, value=tf.concat([strings, values], 0), validate_shape=False)
    return value_tensor, update_op

# TextMetricSpec
def bleu_fn(predictions, labels):
    return _text_metric(predictions, labels, _bleu_fn, 'bleu_metric')
    
def _text_metric(predictions, labels, py_metric_func, scope="text_metric"):
    _separator = ' '
    with tf.variable_scope(scope):
       # Join tokens into single strings
        predictions_flat = tf.reduce_join(
            predictions, 1, separator=_separator)
        labels_flat = tf.reduce_join(
            labels, 1, separator=_separator)

        sources_value, sources_update = accumulate_strings(
           values=predictions_flat, name="sources")
        targets_value, targets_update = accumulate_strings(
           values=labels_flat, name="targets")

        metric_value = tf.py_func(
            func=py_metric_func,
            inp=[sources_value, targets_value],
            Tout=tf.float32,
            name="value")

    with tf.control_dependencies([sources_update, targets_update]):
        update_op = tf.identity(metric_value, name="update_op")

    return metric_value, update_op

def _bleu_fn(hypotheses, references):
    # Deal with byte chars
    if hypotheses.dtype.kind == np.dtype("U"):
        hypotheses = np.char.encode(hypotheses, "utf-8")
    if references.dtype.kind == np.dtype("U"):
        references = np.char.encode(references, "utf-8")

    # Convert back to unicode object
    hypotheses = [_.decode("utf-8") for _ in hypotheses]
    references = [_.decode("utf-8") for _ in references]

    # Slice all hypotheses and references up to SOS -> EOS
    sliced_hypotheses = [utils.slice_text(
        _) for _ in hypotheses]
    sliced_references = [utils.slice_text(
        _) for _ in references]

    bleu_score = bleu.moses_multi_bleu(sliced_hypotheses, sliced_references, lowercase=False) #pylint: disable=E1102
    print('bleu_score:', bleu_score)
    return bleu_score

