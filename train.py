import random
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import math
import threading

# R G B
mean = [124.23002308, 159.76066492, 104.05509866]
std = [47.84116963, 41.94039282, 49.85093766]

cfg = {
    'data_params': {
        'img_shape': (512, 512)
    },
    'model_params': {
        'batchsize_per_gpu': 8,
        'iteration_per_epoch': 128,
        'epoch': 150
    }
}

WIDTH = cfg['data_params']['img_shape'][0]
HEIGHT = cfg['data_params']['img_shape'][1]
SEED = 2021
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_ROOT = "./train_images"
train_img_lists = os.listdir(TRAIN_DATA_ROOT)


# result_mean_R = []
# result_mean_G = []
# result_mean_B = []
# result_std_R = []
# result_std_G = []
# result_std_B = []
#
# for i in range(math.ceil(len(train_img_lists) / 128)):
#     result_mean_R.append([])
#     result_mean_G.append([])
#     result_mean_B.append([])
#     result_std_R.append([])
#     result_std_G.append([])
#     result_std_B.append([])
#
#
# def get_train_set_norm(imglist):
#     for img in imglist:
#         image = Image.open(os.path.join(TRAIN_DATA_ROOT, img))
#         image = image.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
#         r, g, b = image.split()
#         mean_r = np.mean(np.asarray(r).flatten())
#         mean_g = np.mean(np.asarray(g).flatten())
#         mean_b = np.mean(np.asarray(b).flatten())
#         std_r = np.std(np.asarray(r).flatten())
#         std_g = np.std(np.asarray(g).flatten())
#         std_b = np.std(np.asarray(b).flatten())
#         file_name, _ = os.path.splitext(img)
#         with open("./mean_std_information/" + file_name + ".txt", 'w') as writer:
#             writer.write(str(mean_r) + '\n')
#             writer.write(str(mean_g) + '\n')
#             writer.write(str(mean_b) + '\n')
#             writer.write(str(std_r) + '\n')
#             writer.write(str(std_g) + '\n')
#             writer.write(str(std_b) + '\n')
#
#
# # 每次更新WIDTH HEIGHT的时候需要重新运行该函数
# # Enable Multi-Thread
# thread_list = []
# for n in range(math.ceil(len(train_img_lists) / 128)):
#     t1 = threading.Thread(target=get_train_set_norm, args=(
#         train_img_lists[n * 128:min((n + 1) * 128, len(train_img_lists))],))
#     thread_list.append(t1)
# for t in thread_list:
#     t.start()
# for t in thread_list:
#     t.join()


# 已知各组标准差，求总体标准差
# S(All) = sqrt(((S(1)^2+S(2)^2+...+S(n)^2)) +
# ((total_mean - group_mean[1])^2+(total_mean - group_mean[2])^2+...+(total_mean - group_mean[n])^2)
# / len(train_img_lists))


# def read_mean_std():
#     file_list = os.listdir(r"./mean_std_information")
#     data = [[], [], [], [], [], []]
#     for file in file_list:
#         with open('./mean_std_information/' + file) as reader:
#             for i in range(6):
#                 line = reader.readline()
#                 data[i].append(float(line))
#     return data
#
#
# data = read_mean_std()
#
#
# def cal_mean_std(data):
#     mean = np.mean(np.array(data[0:3]), axis=1)
#     std = np.sqrt(
#         (np.sum(np.array(data[3:6]) ** 2, axis=1) + np.array([np.sum((data[0] - mean[0]) ** 2),
#                                                               (np.sum((data[1] - mean[1]) ** 2)),
#                                                               np.sum((data[2] - mean[2]) ** 2)])) / len(
#             train_img_lists))
#     return mean, std
#
#
# mean, std = cal_mean_std(data)
# print(mean)
# print(std)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)
tfrecs = tf.io.gfile.glob("./train_tfrecords/*.tfrecords")
raw_image_dataset = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)

# Create a dictionary describing the features.
image_feature_description = {
    'name': tf.io.FixedLenFeature([], tf.string),
    'data': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_image_function(single_photo):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(single_photo, image_feature_description)


def _decode_image_function(single_photo):
    single_photo['data'] = tf.image.decode_jpeg(single_photo['data'], channels=3)
    return single_photo


def _preprocess_image_function(single_photo):
    image = tf.expand_dims(single_photo['data'], axis=0)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(images=image, size=[HEIGHT, WIDTH])
    i1 = (image[:, :, :, 0] - mean[0] / 255.0) / std[0] * 255.0
    i2 = (image[:, :, :, 1] - mean[1] / 255.0) / std[1] * 255.0
    i3 = (image[:, :, :, 2] - mean[2] / 255.0) / std[2] * 255.0
    # use the train dataset data
    image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=3)
    # image = tf.image.per_image_standardization(image)
    single_photo['data'] = image
    return single_photo


parsed_image_dataset = (raw_image_dataset.map(_parse_image_function)
                        .map(_decode_image_function)
                        .map(_preprocess_image_function))
for image_features in parsed_image_dataset:
    image = image_features['data'].numpy()
    image = Image.fromarray(np.uint8(image[0] * 255.0)).convert('RGB')
    image.show()
