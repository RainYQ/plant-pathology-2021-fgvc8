import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(description="Config TFRecords Generator.")
    parser.add_argument('--train_images', default='./train_images/', type=str)
    parser.add_argument('--number_in_tfrecord', default=128, type=int)
    parser.add_argument('--label', default='./train_without_rep.csv', type=str)
    parser.add_argument('--visualization', default=False, action="store_true")
    parser.add_argument('--njob', default=16, type=int)
    return parser


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


def pares_image(pic_name, mode):
    label = train_data.loc[train_data["image"] == pic_name]["labels"]
    img = Image.open(TRAIN_DATA_ROOT + str(pic_name))
    img = img.convert('RGB')
    img = img.resize((mode, mode), Image.ANTIALIAS)
    img_raw = img.tobytes()
    # 将label存为多标签数组
    return img_raw, label2array[int(label)].tobytes(), label, id2label[int(label)]


def create_dataset(train_data, i, mode):
    with tf.io.TFRecordWriter(r'./tfrecords_' + str(mode) + '/train' + '_' + str(int(i)) + '.tfrecords') as writer:
        for data in tqdm(train_data["image"]):
            raw, labels, label, label_name = pares_image(data, mode)
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
        writer.close()


if __name__ == "__main__":
    if not os.path.exists('./tfrecords_600'):
        os.mkdir('./tfrecords_600')
    if not os.path.exists('./tfrecords_512'):
        os.mkdir('./tfrecords_512')
    parser = get_parser()
    args = parser.parse_args()
    Visualization = args.visualization
    NUMBER_IN_TFRECORD = args.number_in_tfrecord
    TRAIN_DATA_ROOT = args.train_images
    train_data = pd.read_csv(args.label, encoding='utf-8')
    njob = args.njob
    id2label = dict([(value, key) for key, value in label2id.items()])
    train_data["labels"] = train_data["labels"].map(label2id)
    data_count = train_data["labels"].value_counts()
    if Visualization:
        plt.bar([i for i in range(len(data_count))], data_count)
        plt.xlabel('label type')
        plt.ylabel('count')
        plt.show()
    _ = joblib.Parallel(n_jobs=njob)(
        joblib.delayed(create_dataset)
        (train_data[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, train_data.shape[0])], n, 512)
        for n in range(math.ceil(train_data.shape[0] / NUMBER_IN_TFRECORD))
    )
    _ = joblib.Parallel(n_jobs=njob)(
        joblib.delayed(create_dataset)
        (train_data[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, train_data.shape[0])], n, 600)
        for n in range(math.ceil(train_data.shape[0] / NUMBER_IN_TFRECORD))
    )
