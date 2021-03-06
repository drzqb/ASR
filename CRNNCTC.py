import tensorflow as tf
from tensorflow.keras.backend import ctc_decode
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Bidirectional, BatchNormalization, GRU, \
    Activation, TimeDistributed, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
from TFDataUtils import TFDATAUTILS
from CustomLayers import CTCLayer, CTCInputLabelLen

params_epochs = 100
params_lr = 1.0e-3
params_batch_size = 16

params_check = "models/crnnctc/"
params_model_name = "crnnctc.h5"

params_mode = "train1"


class USER():
    def __init__(self):
        self.tfdu = TFDATAUTILS()

    def build_model(self, summary=True):
        audio_input = Input(name='audio_input', shape=(self.tfdu.audio_len, self.tfdu.audio_feature_len, 1),
                            dtype=tf.float32)
        pinyin_labels = Input(name='pinyin_labels', shape=[self.tfdu.label_max_string_len], dtype=tf.int32)

        layer_h1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(audio_input)
        layer_h1 = Dropout(0.1)(layer_h1)
        layer_h2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)
        layer_h3 = MaxPooling2D(pool_size=2, padding="valid")(layer_h2)
        layer_h3 = Dropout(0.1)(layer_h3)

        layer_h4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)
        layer_h6 = MaxPooling2D(pool_size=2, padding="valid")(layer_h5)
        layer_h6 = Dropout(0.1)(layer_h6)

        layer_h7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6)
        layer_h7 = Dropout(0.1)(layer_h7)
        layer_h8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7)
        layer_h9 = MaxPooling2D(pool_size=2, padding="valid")(layer_h8)

        layer_h9 = Dropout(0.1)(layer_h9)
        layer_h9 = BatchNormalization(axis=-1, epsilon=1e-5)(layer_h9)
        layer_h9 = Activation('relu')(layer_h9)
        layer_h10 = TimeDistributed(Flatten(), name='flatten')(layer_h9)

        layer_h11 = Dense(64, activation="relu", name="dense")(layer_h10)
        layer_h11 = Bidirectional(GRU(512, return_sequences=True, implementation=2, dropout=0.1), name='blstm')(
            layer_h11)

        crnnoutput = Dense(self.tfdu.pinyins_len + 2, name='crnnoutput', activation='softmax')(layer_h11)

        # CTC
        input_len, label_len = CTCInputLabelLen(3, name="ctcinputlabellen")(inputs=(audio_input, pinyin_labels))

        predict = CTCLayer(name="ctclayer")(inputs=(pinyin_labels, crnnoutput, input_len, label_len))

        model = Model(inputs=[audio_input, pinyin_labels], outputs=[predict])

        tf.keras.utils.plot_model(model, "crnnctc.jpg", show_shapes=True, dpi=500)

        if summary:
            model.summary(line_length=200)

            # for tv in model.variables:
            #     print(tv.name, " : ", tv.shape)

        return model

    def build_predict_model(self, summary=True):
        audio_input = Input(name='audio_input', shape=(self.tfdu.audio_len, self.tfdu.audio_feature_len, 1),
                            dtype=tf.float32)

        layer_h1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(audio_input)
        layer_h1 = Dropout(0.1)(layer_h1)
        layer_h2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)
        layer_h3 = MaxPooling2D(pool_size=2, padding="valid")(layer_h2)
        layer_h3 = Dropout(0.1)(layer_h3)

        layer_h4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)
        layer_h6 = MaxPooling2D(pool_size=2, padding="valid")(layer_h5)
        layer_h6 = Dropout(0.1)(layer_h6)

        layer_h7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6)
        layer_h7 = Dropout(0.1)(layer_h7)
        layer_h8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7)
        layer_h9 = MaxPooling2D(pool_size=2, padding="valid")(layer_h8)

        layer_h9 = Dropout(0.1)(layer_h9)
        layer_h9 = BatchNormalization(axis=-1, epsilon=1e-5)(layer_h9)
        layer_h9 = Activation('relu')(layer_h9)
        layer_h10 = TimeDistributed(Flatten(), name='flatten')(layer_h9)

        layer_h11 = Dense(64, activation="relu", name="dense")(layer_h10)
        layer_h11 = Bidirectional(GRU(512, return_sequences=True, implementation=2, dropout=0.1), name='blstm')(
            layer_h11)

        crnnoutput = Dense(self.tfdu.pinyins_len + 2, name='crnnoutput', activation='softmax')(layer_h11)

        model = Model(inputs=[audio_input], outputs=[crnnoutput])

        if summary:
            model.summary()

        return model

    def train(self):
        train_dataset = self.tfdu.batched_data("data/TFRecordFiles/thchs30_train.tfrecord",
                                               self.tfdu.single_example_parser,
                                               params_batch_size,
                                               padded_shapes=(([self.tfdu.audio_len, self.tfdu.audio_feature_len],
                                                               [self.tfdu.label_max_string_len]),
                                                              [self.tfdu.label_max_string_len]))

        dev_dataset = self.tfdu.batched_data("data/TFRecordFiles/thchs30_dev.tfrecord",
                                             self.tfdu.single_example_parser,
                                             params_batch_size,
                                             padded_shapes=(([self.tfdu.audio_len, self.tfdu.audio_feature_len],
                                                             [self.tfdu.label_max_string_len]),
                                                            [self.tfdu.label_max_string_len]))

        model = self.build_model()
        if params_mode == "train1":
            model.load_weights(params_check + params_model_name)

        optimizer = Adam(params_lr)
        model.compile(optimizer)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint(filepath=params_check + params_model_name,
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True)
        ]

        model.fit(
            train_dataset,
            epochs=params_epochs,
            validation_data=dev_dataset,
            callbacks=callbacks
        )

        model.save_weights(params_check + params_model_name)

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = ctc_decode(pred, input_length=input_len, greedy=True, beam_width=self.tfdu.label_max_string_len)[0][0]
        # Iterate over the results and get back the pinyin
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.tfdu.num_to_pinyin(res), separator=" ").numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def test(self, audois):
        model = self.build_predict_model(summary=False)
        model.load_weights(params_check + params_model_name)

        m_samples = len(audois)
        audioinput = np.zeros([m_samples, self.tfdu.audio_len, self.tfdu.audio_feature_len], dtype=np.float)

        for i in range(m_samples):
            res = self.tfdu.encode_single_sample(audois[i])
            audio = res["wav"]
            audio_len = len(audio)
            audioinput[i, :audio_len] = audio

        preds = model.predict(audioinput)
        pred_texts = self.decode_batch_predictions(preds)

        for i in range(m_samples):
            predlabel = pred_texts[i]
            print("predict: ", predlabel)


def main():
    if not os.path.exists(params_check):
        os.makedirs(params_check)

    user = USER()

    if params_mode.startswith('train'):
        user.train()

    elif params_mode == 'test':
        audios = [
            "data/data_thchs30/test/D4_750.wav",
        ]
        user.test(audios)


if __name__ == "__main__":
    main()
