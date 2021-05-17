# -*- coding: utf-8 -*-

"""
Extra Datset TFRecords Generator:
- Download Dataset in plant-pathology-2020-fgvc7
- Move all 'Test_' images to ./plant-pathology-2020-fgvc7/test
- Move all 'Train_' images to ./plant-pathology-2020-fgvc7/train
- Make dir './train_extra_tfrecords'
Modify:
    - Line 73-78 可能需要限制多线程并发数量
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import threading
import math
from PIL import Image
from Preprocess import id2label, label2array, label2id

NUMBER_IN_TFRECORD = 128
TRAIN_DATA_ROOT = "./plant-pathology-2020-fgvc7/train/"

classes = np.array([
    'healthy',
    'multiple_diseases',
    'rust',
    'scab'])

extra_train_data = pd.read_csv("./plant-pathology-2020-fgvc7/train.csv", encoding='utf-8')
image_name = [id + ".jpg" for id in list(extra_train_data['image_id'])]
labels = []
for indexs in extra_train_data.index:
    label = ' '.join(classes[np.array(list(extra_train_data.loc[indexs].values[1:])).astype('bool')])
    labels.append(label)
extra_data_table = pd.DataFrame({'image': image_name, 'labels': labels})
extra_data_table = extra_data_table.loc[extra_data_table["labels"] != 'multiple_diseases']
print("extra label type:", len(set(extra_data_table["labels"])))
extra_data_count = extra_data_table["labels"].value_counts()
print(extra_data_count)
extra_data_table["labels"] = extra_data_table["labels"].map(label2id)


def pares_image(pic_name):
    label = extra_data_table.loc[extra_data_table["image"] == pic_name]["labels"]
    img = Image.open(TRAIN_DATA_ROOT + str(pic_name))
    img = img.convert('RGB')
    img = img.resize((512, 512), Image.ANTIALIAS)
    img_raw = img.tobytes()
    return img_raw, label2array[int(label)].tobytes(), label, id2label[int(label)]


def create_dataset(train_data, i):
    with tf.io.TFRecordWriter(r'./plant-pathology-2020-fgvc7/train_extra_tfrecords/train_extra' + '_' + str(int(i)) + '.tfrecords') as writer:
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
    for n in range(math.ceil(extra_data_table.shape[0] / NUMBER_IN_TFRECORD)):
        t1 = threading.Thread(target=create_dataset, args=(
            extra_data_table[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, extra_data_table.shape[0])], n))
        t1.start()
