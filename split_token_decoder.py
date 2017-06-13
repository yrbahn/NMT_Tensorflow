import tensorflow as tf

"""
SplitTokensDecoder Class

"""
class SplitTokensDecoder(tf.contrib.slim.data_decoder.DataDecoder):
    """문자열 텐서를 토근들로 분리하고 그 토근들과 길이를 리턴한다.
    선택적으로 특별한 토근을 앞에 추가하거나 뒤에 넣을수 있다.
    
    인수들:
        delimiter
        tokens_features_name
        length_features_naem
    """
    def __init__(self,
                 delimiter=" ", 
                 tokens_feature_name="tokens",
                 length_feature_name="length",
                 prepend_token=None,
                 append_token=None):
        self.delimiter = delimiter
        self.tokens_feature_name = tokens_feature_name
        self.length_feature_name = length_feature_name
        self.prepend_token = prepend_token
        self.append_token = append_token

    def decode(self, data, items):
        decoded_items = {}

        # 토근으로 분리하기
        tokens = tf.string_split([data], delimiter=self.delimiter).values

        # Optionally prepend a special token
        if self.prepend_token is not None:
            tokens = tf.concat([[self.prepend_token], tokens], 0)

        # Optionally append a special token
        if self.append_token is not None:
            tokens = tf.concat([tokens, [self.append_token]], 0)

        decoded_items[self.length_feature_name] = tf.size(tokens)
        decoded_items[self.tokens_feature_name] = tokens
        return [decoded_items[_] for _ in items]

    def list_items(self):
        return [self.tokens_feature_name, self.length_feature_name]
