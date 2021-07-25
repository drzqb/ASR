import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import ctc_batch_cost


class CTCLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.loss_fn = ctc_batch_cost

    def call(self, inputs, **kwargs):
        y_true, y_pred, input_len, label_len = inputs

        loss = self.loss_fn(y_true, y_pred, input_len, label_len)

        # self.add_loss(tf.reduce_mean(loss))
        self.add_loss(loss)

        return y_pred


class CTCInputLabelLen(Layer):
    def __init__(self, count, **kwargs):
        super(CTCInputLabelLen, self).__init__(**kwargs)

        self.count = count

    def call(self, inputs, **kwargs):
        audio_input, pinyin_labels = inputs

        input_len = tf.reduce_sum(tf.cast(tf.greater(audio_input, 0), tf.int32), axis=1)[:, 0]
        for _ in range(self.count):
            input_len = tf.math.ceil(input_len / 2)

        input_len = tf.cast(input_len, tf.int32)

        label_len = tf.reduce_sum(tf.cast(tf.greater(pinyin_labels, 0), tf.int32), axis=1, keepdims=True)

        return input_len, label_len
