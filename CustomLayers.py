import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import ctc_batch_cost


class CTCLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.loss_fn = ctc_batch_cost

    def call(self, inputs, **kwargs):
        y_true, y_pred = inputs
        batch_len, label_len = tf.shape(y_true)[0], tf.shape(y_true)[1]
        input_len = tf.shape(y_pred)[1]

        input_length = input_len * tf.ones(shape=(batch_len, 1), dtype=tf.int32)
        label_length = label_len * tf.ones(shape=(batch_len, 1), dtype=tf.int32)

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        # self.add_loss(tf.reduce_mean(loss))
        self.add_loss(loss)

        return y_pred
