import tensorflow as tf
import numpy as np
import wave
from scipy.fftpack import fft
from pathlib import Path
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tqdm import tqdm


class TFDATAUTILS():
    def __init__(self):
        self.pinyin = []
        with open("data/pinyin.dict", "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                py = line.split("\t")[0]
                self.pinyin.append(py)

        self.pinyin = self.pinyin

        self.pinyins_len = len(self.pinyin)

        self.pinyin_to_num = StringLookup(vocabulary=self.pinyin)
        self.num_to_pinyin = StringLookup(vocabulary=self.pinyin_to_num.get_vocabulary(), invert=True)

        self.audio_len = 1600
        self.audio_feature_len = 200
        self.label_max_string_len = 64

        x = np.linspace(0, self.audio_feature_len * 2 - 1, self.audio_feature_len * 2, dtype=np.int64)
        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * x / (self.audio_feature_len * 2 - 1))  # 汉明窗

    @staticmethod
    def read_wav_data(filename):
        '''
        读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
        '''
        with wave.open(filename, "rb") as wav:
            num_frame = wav.getnframes()  # 获取帧数
            num_channel = wav.getnchannels()  # 获取声道数
            framerate = wav.getframerate()  # 获取帧速率
            str_data = wav.readframes(num_frame)  # 读取全部的帧

        wave_data = np.frombuffer(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
        wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
        wave_data = wave_data.T  # 将矩阵转置

        return wave_data, framerate

    def GetFrequencyFeature3(self, wavsignal, fs):
        # wav波形 加时间窗以及时移10ms
        time_window = 25  # 单位ms

        wav_arr = np.array(wavsignal)
        wav_length = wav_arr.shape[1]

        range0_end = int(len(wavsignal[0]) / fs * 1000 - time_window) // 10  # 计算循环终止的位置，也就是最终生成的窗数
        data_input = np.zeros((range0_end, self.audio_feature_len), dtype=np.float)  # 用于存放最终的频率特征数据

        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + self.audio_feature_len * 2

            data_line = wav_arr[0, p_start:p_end]

            data_line = data_line * self.w  # 加窗

            data_line = np.abs(fft(data_line)) / wav_length

            data_input[i] = data_line[0:self.audio_feature_len]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的

        data_input = np.log(data_input + 1)

        return data_input

    def encode_single_sample(self, wav_path, label_path=None):
        wave_data, fs = self.read_wav_data(wav_path)
        data_input = self.GetFrequencyFeature3(wave_data, fs)

        if label_path is not None:
            with open(label_path, "r", encoding="utf-8") as fr:
                k = 0
                for line in fr:
                    if k < 1:
                        k += 1
                        continue

                    line = line.strip().split(" ")
                    label = self.pinyin_to_num(line)

                    return {"wav": data_input, "label": label}
        else:
            return {"wav": data_input}

    def maketfrecord(self, datapath, tfrecordfile, mode="train"):
        data_dir = Path(datapath)

        writer = tf.io.TFRecordWriter(tfrecordfile)
        wav = sorted(list(map(str, list(data_dir.glob("*.wav")))))
        labels = [p.replace(mode, "data") + ".trn" for p in wav]

        m_samples = 0

        for w, l in tqdm(zip(wav, labels)):
            res = self.encode_single_sample(w, l)
            wav_data = res["wav"]
            lab_data = res["label"]

            wav_feature = [tf.train.Feature(float_list=tf.train.FloatList(value=[wav_])) for wav_ in
                           wav_data.reshape([-1])]
            lab_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[lab_])) for lab_ in lab_data]

            seq_example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(feature_list={
                    'wav': tf.train.FeatureList(feature=wav_feature),
                    'lab': tf.train.FeatureList(feature=lab_feature),
                })
            )
            serialized = seq_example.SerializeToString()
            writer.write(serialized)

            m_samples += 1

        print("样本数：", m_samples)

    def single_example_parser(self, serialized_example):
        sequence_features = {
            'wav': tf.io.FixedLenSequenceFeature([], tf.float32),
            'lab': tf.io.FixedLenSequenceFeature([], tf.int64)
        }

        _, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized=serialized_example,
            sequence_features=sequence_features
        )

        wav = tf.reshape(tf.cast(sequence_parsed['wav'], tf.float32), [-1, self.audio_feature_len])
        lab = tf.cast(sequence_parsed['lab'], tf.int32)

        return (wav, lab), lab

    @staticmethod
    def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, shuffle=True):
        dataset = tf.data.TFRecordDataset(tfrecord_filename)
        if shuffle:
            dataset = dataset.shuffle(100 * batch_size)

        dataset = dataset.map(single_example_parser) \
            .padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=False) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == "__main__":
    tfdu = TFDATAUTILS()
    # tfdu.maketfrecord("data/data_thchs30/test/", "data/TFRecordFiles/thchs30_test.tfrecord", mode="test")
    res=tfdu.encode_single_sample("data/data_thchs30/train/A11_0.wav","data/data_thchs30/data/A11_0.wav.trn")
    print(res["wav"].shape)
    print(res["label"].shape)