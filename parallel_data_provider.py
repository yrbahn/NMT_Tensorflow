import tensorflow as tf
import numpy as np

from tensorflow.contrib.slim.python.slim.data import data_provider
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from split_token_decoder import SplitTokensDecoder

"""
ParallelDataProvider class
"""
class ParallelDataProvider(data_provider.DataProvider):
    """
    인수:
        data
    """
    def __init__(self,
                dataset1,
                dataset2,
                shuffle=True,
                num_epochs=None,
                common_queue_capacity=4096,
                common_queue_min=1024,
                seed=None):
        if seed is None:
            seed = np.random.randint(10e8)
        
        _, data_source = parallel_reader.parallel_read(
            dataset1.data_sources,
            reader_class=dataset1.reader,
            num_readers=1, # 생성할 Reader객체 수
            shuffle=False,
            capacity=common_queue_min,
            min_after_dequeue=common_queue_min,
            seed=seed)
        
        data_target = ""
        if dataset2 is not None:
            _, data_target = parallel_reader.parallel_read(
                dataset2.data_sources,
                reader_class=dataset2.reader,
                num_epochs=num_epochs,
                num_readers=1,
                shuffle=False,
                capacity=common_queue_capacity,
                min_after_dequeue=common_queue_min,
                seed=seed)
        
        # 옵션으로 데이타를 셔플한다.
        if shuffle:
            shuffle_queue = tf.RandomShuffleQueue(
                capacity=common_queue_capacity,
                min_after_dequeue=common_queue_min,
                dtypes=[tf.string, tf.string],
                seed=seed)
            
            enqueue_ops = []
            enqueue_ops.append(shuffle_queue.enqueue([data_source, data_target]))
            tf.train.add_queue_runner(
                tf.train.QueueRunner(shuffle_queue, enqueue_ops))
            data_source, data_target = shuffle_queue.dequeue()

            
        # Decode source items
        items = dataset1.decoder.list_items()
        tensors = dataset1.decoder.decode(data_source, items)

        if dataset2 is not None:
            # Decode target items
            items2 = dataset2.decoder.list_items()
            tensors2 = dataset2.decoder.decode(data_target, items2)

        # Merge items and results
        items = items + items2
        tensors = tensors + tensors2

        super(ParallelDataProvider, self).__init__(
            items_to_tensors=dict(zip(items, tensors)),
            num_samples=dataset1.num_samples)

# make parallel data provider
def make_parallel_data_provider(data_sources_source,
                               data_sources_target,
                               reader=tf.TextLineReader,
                               num_samples=None,
                               source_delimiter=" ",
                               target_delimiter=" ",
                               **kwargs):
    #data_sources_source : 소스 텍스트 파일에 data source 리스트
    #data_sources_target : 타켓 텍스트 파일에 data source 리스트
    #num_samples: 선택사항, 데이타 셋에 레코드 수
    
    decoder_source = SplitTokensDecoder(
        tokens_feature_name="source_tokens",
        length_feature_name="source_len",
        append_token="SEQUENCE_END",
        delimiter=source_delimiter)
    
    dataset_source = tf.contrib.slim.dataset.Dataset(
        data_sources=data_sources_source,
        reader=reader,
        decoder=decoder_source,
        num_samples=num_samples,
        items_to_descriptions={})

    dataset_target = None
    if data_sources_target is not None:
        decoder_target = SplitTokensDecoder(
            tokens_feature_name="target_tokens",
            length_feature_name="target_len",
            prepend_token="SEQUENCE_START",
            append_token="SEQUENCE_END",
            delimiter=target_delimiter)

        dataset_target = tf.contrib.slim.dataset.Dataset(
            data_sources=data_sources_target,
            reader=reader,
            decoder=decoder_target,
            num_samples=num_samples,
            items_to_descriptions={})

    return ParallelDataProvider(
        dataset1=dataset_source, dataset2=dataset_target, **kwargs)
