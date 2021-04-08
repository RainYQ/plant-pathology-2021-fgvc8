# -*- coding: utf-8 -*-

"""
TFRecords Generator:
Modify:
    - Line 113-116 可能需要限制多线程并发数量
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import math

Visualization = False
NUMBER_IN_TFRECORD = 128
TRAIN_DATA_ROOT = "./train_images"
TEST_DATA_ROOT = "./test_images"
train_data = pd.read_csv("./train_without_rep.csv", encoding='utf-8')
test_data = os.listdir(TEST_DATA_ROOT)
print("label type:", len(set(train_data["labels"])))

label2id = {
    'scab': 0,
    'healthy': 1,
    'frog_eye_leaf_spot': 2,
    'rust': 3,
    'complex': 4,
    'powdery_mildew': 5,
    'scab frog_eye_leaf_spot': 6,
    'scab frog_eye_leaf_spot complex': 7,
    'frog_eye_leaf_spot complex': 8,
    'rust frog_eye_leaf_spot': 9,
    'rust complex': 10,
    'powdery_mildew complex': 11
}

label2array = {
    0: np.array([1, 0, 0, 0, 0], dtype=np.float32),
    1: np.array([0, 0, 0, 0, 0], dtype=np.float32),
    2: np.array([0, 1, 0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 1, 0, 0], dtype=np.float32),
    4: np.array([0, 0, 0, 1, 0], dtype=np.float32),
    5: np.array([0, 0, 0, 0, 1], dtype=np.float32),
    6: np.array([1, 1, 0, 0, 0], dtype=np.float32),
    7: np.array([1, 1, 0, 1, 0], dtype=np.float32),
    8: np.array([0, 1, 0, 1, 0], dtype=np.float32),
    9: np.array([0, 1, 1, 0, 0], dtype=np.float32),
    10: np.array([0, 0, 1, 1, 0], dtype=np.float32),
    11: np.array([0, 0, 0, 1, 1], dtype=np.float32)
}

# id2label用于输入0-11, 查找label原始名称
id2label = dict([(value, key) for key, value in label2id.items()])
# print(id2label)
# 替换label到0-11
train_data["labels"] = train_data["labels"].map(label2id)
# 统计不同label的数量
# 属于严重不均匀多分类问题
data_count = train_data["labels"].value_counts()
# 绘制直方图
if Visualization:
    plt.bar([i for i in range(12)], data_count)
    plt.xlabel('label type')
    plt.ylabel('count')
    plt.show()


def pares_image(pic_name):
    label = train_data.loc[train_data["image"] == pic_name]["labels"]
    # Using Tensorflow io
    img_raw = open("./train_images/" + str(pic_name), 'rb').read()
    img_shape = tf.image.decode_jpeg(img_raw).shape
    # Using Image
    # img = Image.open("./train_images/" + str(pic_name))
    # img = img.convert('RGB')
    # img = img.resize((512, 512), Image.ANTIALIAS)
    # img_raw = img.tobytes()
    return img_raw, label2array[int(label)].tobytes(), label, id2label[int(label)]


def create_dataset(train_data, i):
    with tf.io.TFRecordWriter(r'./train_tfrecords/train' + '_' + str(int(i)) + '.tfrecords') as writer:
        for data in tqdm(train_data["image"]):
            raw, labels, label, label_name = pares_image(data)
            exam = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.encode('utf-8')])),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])),
                        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                        'label_name': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[label_name.encode('utf-8')]))
                    }
                )
            )
            writer.write(exam.SerializeToString())


if __name__ == "__main__":
    # Enable Multi-Thread
    for n in range(math.ceil(train_data.shape[0] / NUMBER_IN_TFRECORD)):
        t1 = threading.Thread(target=create_dataset, args=(
            train_data[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, train_data.shape[0])], n))
        t1.start()
