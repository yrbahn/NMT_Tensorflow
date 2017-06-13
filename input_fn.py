import tensorflow as tf
from parallel_data_provider import make_parallel_data_provider

FEATURE_KEY = set(["source_tokens", "source_len"])
LABEL_KEY   = set(["target_tokens", "target_len"])

def create_input_fn(source_file_list, 
                    target_file_list,
                    batch_size,
                    bucket_boundaries=None,
                    allow_smaller_final_batch=False,
                    scope=None):
    def input_fn():
        with tf.variable_scope(scope or "input_fn"):
            data_provider = make_parallel_data_provider(data_sources_source=source_file_list,
                                                        data_sources_target=target_file_list,
                                                        #num_epochs=num_epochs,
                                                        shuffle=True)
            
            #data_provider.get(item_list) => tensor_list 
            #data_provider.list_items() => 아이템 이름 리스트
            item_values = data_provider.get(list(data_provider.list_items()))
            features_and_labels = dict(zip(data_provider.list_items(), item_values))
            
            
            if bucket_boundaries:
                _, batch = tf.contrib.training.bucket_by_sequence_length(
                    input_length=features_and_labels["source_len"],
                    bucket_boundaries=bucket_boundaries,
                    tensors=features_and_labels,
                    batch_size=batch_size,
                    keep_input=features_and_labels["source_len"] >= 1,
                    dynamic_pad=True,
                    capacity=5000 + 16 * batch_size,
                    allow_smaller_final_batch=allow_smaller_final_batch,
                    name="bucket_queue")
            else:
                batch = tf.train.batch(
                    tensors=features_and_labels,
                    enqueue_many=False,
                    batch_size=batch_size,
                    dynamic_pad=True,
                    capacity=5000 + 16 * batch_size,
                    allow_smaller_final_batch=allow_smaller_final_batch,
                    name="batch_queue")

            # 피쳐와 라벨을 분리하자
            features_batch = {k: batch[k] for k in FEATURE_KEY}
            if set(batch.keys()).intersection(LABEL_KEY):
                labels_batch = {k: batch[k] for k in LABEL_KEY}
            else:
                labels_batch = None
            
            return features_batch, labels_batch

    return input_fn

