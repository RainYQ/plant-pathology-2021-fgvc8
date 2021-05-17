# -*- coding: utf-8 -*-

"""
Extra Datset TFRecords Generator:
- Download Dataset in plant-pathology-2020-fgvc7
- Move all 'Test_' images to ./plant-pathology-2020-fgvc7/test
- Move all 'Train_' images to ./plant-pathology-2020-fgvc7/train
- Make dir './train_extra_tfrecords'
- Make dir './extra_tfrecords'
- Move all images to ./plant-pathology-2020-fgvc7/all
Modify:
    - Line 65-70 可能需要限制多线程并发数量
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import threading
import math
from PIL import Image
import os
from config import cfg

NUMBER_IN_TFRECORD = 128
USE_TTA = False
TRAIN_DATA_ROOT = "./plant-pathology-2020-fgvc7/all/"
if cfg['model_params']['random_resize'] & USE_TTA:
    TFRECORDS_LOCATION = './plant-pathology-2020-fgvc7/extra_tfrecords_600/test_extra'
else:
    TFRECORDS_LOCATION = './plant-pathology-2020-fgvc7/extra_tfrecords_512/test_extra'

HEIGHT_T, WIDTH_T = cfg['data_params']['test_img_shape']
HEIGHT_LARGE, WIDTH_LARGE = cfg['data_params']['over_bound_img_shape']

image_names = os.listdir(TRAIN_DATA_ROOT)


def pares_image(pic_name):
    img = Image.open(TRAIN_DATA_ROOT + str(pic_name))
    img = img.convert('RGB')
    if cfg['model_params']['random_resize'] & USE_TTA:
        img = img.resize((HEIGHT_LARGE, WIDTH_LARGE), Image.ANTIALIAS)
    else:
        img = img.resize((HEIGHT_T, WIDTH_T), Image.ANTIALIAS)
    img_raw = img.tobytes()
    return img_raw


def create_dataset(train_data, i):
    with tf.io.TFRecordWriter(TFRECORDS_LOCATION + '_' + str(int(i)) + '.tfrecords') as writer:
        for data in tqdm(train_data):
            raw = pares_image(data)
            exam = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.encode('utf-8')])),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])),
                    }
                )
            )
            writer.write(exam.SerializeToString())


if __name__ == "__main__":
    # Enable Multi-Thread
    for n in range(math.ceil(len(image_names) / NUMBER_IN_TFRECORD)):
        t1 = threading.Thread(target=create_dataset, args=(
            image_names[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, len(image_names))], n))
        t1.start()
