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
from config import classes, cfg

NUMBER_IN_TFRECORD = 128
CLASS_N = 5
WIDTH = cfg['data_params']['img_shape'][0]
HEIGHT = cfg['data_params']['img_shape'][1]
WIDTH_LARGE = cfg['data_params']['over_bound_img_shape'][0]
HEIGHT_LARGE = cfg['data_params']['over_bound_img_shape'][1]
TRAIN_DATA_ROOT = "./plant-pathology-2020-fgvc7/all"

train_data = os.listdir(TRAIN_DATA_ROOT)
USE_TTA = False
# if cfg['model_params']['random_resize'] & USE_TTA:
#     TFRECORDS_PATH = './plant-pathology-2020-fgvc7/tfrecords/tfrecords_RainYQ_600/train_extra'
# else:
#     TFRECORDS_PATH = './plant-pathology-2020-fgvc7/tfrecords/tfrecords_RainYQ_512/train_extra'
TFRECORDS_PATH = './plant-pathology-2020-fgvc7/tfrecords/tfrecords_noresize/train_extra'
pseudo_data = pd.read_csv("./plant-pathology-2020-fgvc7/pseudo_data.csv", encoding='utf-8')


def pares_image(pic_name):
    label = np.array(pseudo_data.loc[pseudo_data["image"] == pic_name].iloc[:, 1:6], dtype=np.float32)[0]
    # image = tf.io.read_file("./plant-pathology-2020-fgvc7/all/" + str(pic_name))
    # image = tf.image.decode_jpeg(image, channels=3)
    # tf.image.convert_image_dtype(image, tf.float32)
    # if cfg['model_params']['random_resize'] & USE_TTA:
    #     image = tf.image.resize(image, [HEIGHT_LARGE, WIDTH_LARGE])
    # else:
    #     image = tf.image.resize(image, [HEIGHT, WIDTH])
    # image = tf.image.convert_image_dtype(image, saturate=True, dtype=tf.uint8)
    image = open("./plant-pathology-2020-fgvc7/all/" + str(pic_name), 'rb').read()
    label_name = ' '.join(classes[np.round(label).astype('bool')])
    # 12 is out range of train data
    return image, label.tobytes(), 12, label_name


def create_dataset(train_data, i):
    with tf.io.TFRecordWriter(TFRECORDS_PATH + '_' + str(int(i)) + '.tfrecords') as writer:
        for data in tqdm(train_data):
            raw, labels, label, label_name = pares_image(data)
            exam = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.encode('utf-8')])),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])),
                        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'label_name': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[label_name.encode('utf-8')]))
                    }
                )
            )
            writer.write(exam.SerializeToString())


if __name__ == "__main__":
    # Enable Multi-Thread
    for n in range(math.ceil(len(train_data) / NUMBER_IN_TFRECORD)):
        t1 = threading.Thread(target=create_dataset, args=(
            train_data[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, len(train_data))], n))
        t1.start()
