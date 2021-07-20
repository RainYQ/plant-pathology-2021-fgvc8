import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import math
import threading
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Config TFRecords Generator.")
    parser.add_argument('--extra_train_images', default='./plant-pathology-2020-fgvc7/all/', type=str)
    parser.add_argument('--number_in_tfrecord', default=128, type=int)
    parser.add_argument('--pseudo_labels', default='./pseudo_data.csv', type=str)
    return parser


classes = np.array([
    'scab',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew'])


def pares_image(pic_name, size, TRAIN_DATA_ROOT):
    label = np.array(pseudo_data.loc[pseudo_data["image"] == pic_name].iloc[:, 1:6], dtype=np.float32)[0]
    image = tf.io.read_file(TRAIN_DATA_ROOT + str(pic_name))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [size, size])
    image = tf.image.convert_image_dtype(image, saturate=True, dtype=tf.uint8)
    label_name = ' '.join(classes[np.round(label).astype('bool')])
    return image.numpy().tobytes(), label.tobytes(), 12, label_name


def create_dataset(train_data, i, size, TRAIN_DATA_ROOT):
    with tf.io.TFRecordWriter(
            './tfrecords_' + str(size) + '_extra/train_extra' + '_' + str(int(i)) + '.tfrecords') as writer:
        for data in tqdm(train_data):
            raw, labels, label, label_name = pares_image(data, size, TRAIN_DATA_ROOT)
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
    if not os.path.exists('./tfrecords_600_extra'):
        os.mkdir('./tfrecords_600_extra')
    if not os.path.exists('./tfrecords_512_extra'):
        os.mkdir('./tfrecords_512_extra')
    parser = get_parser()
    args = parser.parse_args()
    NUMBER_IN_TFRECORD = args.number_in_tfrecord
    CLASS_N = 5
    TRAIN_DATA_ROOT = args.extra_train_images
    pseudo_data = pd.read_csv(args.pseudo_labels, encoding='utf-8')
    train_data = os.listdir(TRAIN_DATA_ROOT)
    for n in range(math.ceil(len(train_data) / NUMBER_IN_TFRECORD)):
        t1 = threading.Thread(target=create_dataset, args=(
            train_data[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, len(train_data))], n, 512,
            TRAIN_DATA_ROOT))
        t1.start()
    for n in range(math.ceil(len(train_data) / NUMBER_IN_TFRECORD)):
        t1 = threading.Thread(target=create_dataset, args=(
            train_data[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, len(train_data))], n, 600,
            TRAIN_DATA_ROOT))
        t1.start()
