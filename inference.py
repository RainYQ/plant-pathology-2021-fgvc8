"""
Inference on k-fold Model:
Modify:
    - Line 23 K
    - Line 25 USE_TTA
    - Line 28 TTA_STEP
    - Line 222 model location
"""

import tensorflow as tf
from GroupNormalization import GroupNormalization
import efficientnet.tfkeras as efn
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import math
import random
import tensorflow_addons as tfa
from config import classes, cfg

# k-fold number
K = 5
# TTA(测试时增强)
USE_TTA = True
# TTA增强测试次数
if USE_TTA:
    TTA_STEP = 8
else:
    TTA_STEP = 1

# SEED = 2021

WITH_OUT_LABLE = True

mean = [124.23002308, 159.76066492, 104.05509866]
std = [47.84116963, 41.94039282, 49.85093766]

AUTOTUNE = tf.data.experimental.AUTOTUNE

CLASS_N = cfg['data_params']['class_type']


# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)


# seed_everything(SEED)

probability = [5704, 4350, 2027, 2124, 1271]
probability = np.array(probability, dtype=np.float32) / sum(probability)
print(probability)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

HEIGHT_T, WIDTH_T = cfg['data_params']['test_img_shape']
HEIGHT_LARGE, WIDTH_LARGE = cfg['data_params']['over_bound_img_shape']
TEST_DATA_ROOT = "./plant-pathology-2020-fgvc7/all"

image_feature_description = {
    'name': tf.io.FixedLenFeature([], tf.string),
    'data': tf.io.FixedLenFeature([], tf.string)
}


def _parse_sample(single_photo):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(single_photo, image_feature_description)


def _preprocess_image_test_function(single_photo):
    image = tf.io.decode_raw(single_photo['data'], tf.uint8)
    name = single_photo['name']
    if cfg['model_params']['random_resize'] & USE_TTA:
        image = tf.reshape(image, [HEIGHT_LARGE, WIDTH_LARGE, 3])
    else:
        image = tf.reshape(image, [HEIGHT_T, WIDTH_T, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    if cfg['model_params']['standardization']:
        i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
        i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
        i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
        image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)],
                          axis=2)
    if USE_TTA:
        # image = tf.image.random_jpeg_quality(image, 80, 100)
        # # 高斯噪声的标准差为 0.3
        # gau = tf.keras.layers.GaussianNoise(0.3)
        # # 以 50％ 的概率为图像添加高斯噪声
        # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
        image = tf.image.random_contrast(image, lower=1.0, upper=1.3)
        # image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        # # brightness随机调整
        # image = tf.image.random_brightness(image, 0.3)
        # random left right flip
        image = tf.image.random_flip_left_right(image)
        # random up down flip
        image = tf.image.random_flip_up_down(image)
        # # cutout ~2 patches / image
        # # width / height 20
        # image = tf.expand_dims(image, axis=0)
        # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
        # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
        # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
        # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
        # image = tf.squeeze(image, axis=0)
        # 随机旋转图片 -30° ~ 30°
        angle = tf.random.uniform([], minval=-np.pi/6, maxval=np.pi/6)
        image = tfa.image.rotate(image, angle)
        # image = tf.expand_dims(image, axis=0)
        # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
        # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
        # image = tf.squeeze(image, axis=0)
        if cfg['model_params']['random_resize'] & USE_TTA:
            image = tf.image.random_crop(image, [HEIGHT_T, WIDTH_T, 3])
    return image, name


if cfg['model_params']['random_resize'] & USE_TTA:
    filenames = tf.io.gfile.glob("./extra_tfrecords_600/*.tfrecords")
else:
    filenames = tf.io.gfile.glob("./extra_tfrecords_512/*.tfrecords")
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False
tdataset = (tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
            .with_options(ignore_order)
            .map(_parse_sample, num_parallel_calls=AUTOTUNE).cache()
            .map(_preprocess_image_test_function, num_parallel_calls=AUTOTUNE)
            .batch(cfg['model_params']['batchsize_in_test']).prefetch(AUTOTUNE))


def create_test_model():
    # backbone = efn.EfficientNetB7(
    #     include_top=False,
    #     input_shape=(HEIGHT_T, WIDTH_T, 3),
    #     weights=None,
    #     pooling='avg'
    # )

    backbone = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        input_shape=(HEIGHT_T, WIDTH_T, 3),
        weights=None,
        pooling='avg'
    )

    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    return model


def inference(count, path):
    model.load_weights(path + "/model_best_%d.h5" % count)
    rec_ids = []
    probs = []
    for data, name in tqdm(tdataset):
        pred = model.predict_on_batch(tf.reshape(data, [-1, HEIGHT_T, WIDTH_T, 3]))
        rec_id_stack = tf.reshape(name, [-1, 1])
        for rec in name.numpy():
            assert len(np.unique(rec)) == 1
        rec_ids.append(rec_id_stack.numpy()[:, 0])
        probs.append(pred)
    crec_ids = np.concatenate(rec_ids)
    cprobs = np.concatenate(probs)
    sub_with_prob = pd.DataFrame({
        'name': list(map(lambda x: x.decode(), crec_ids.tolist())),
        **{classes[i]: cprobs[:, i] / (K * TTA_STEP) for i in range(CLASS_N)}
    })
    sub_with_prob = sub_with_prob.sort_values('name')
    return sub_with_prob


model = create_test_model()
model.summary()


def submission_writer(path, USE_PROBABILITY):
    sub_with_prob = sum(
        map(
            lambda j:
            inference(math.floor(j / TTA_STEP), path).set_index('name'), range(K * TTA_STEP)
        )
    ).reset_index()

    labels = []
    names = []
    for index, row in sub_with_prob.iterrows():
        names.append(row[0])
        probs = np.array(row[1:CLASS_N + 1], dtype=np.float32)
        if USE_PROBABILITY:
            prob = probs > probability
        else:
            prob = np.around(probs)
        prob = prob.astype('bool')
        label = ' '.join(classes[prob])
        # 很重要，视为疾病检测模型，没有检测到疾病时视为健康
        if label == '':
            label = 'healthy'
        labels.append(label)

    sub = pd.DataFrame({
        'image': names,
        'labels': labels})
    sub.to_csv('submission.csv', index=False)
    sub_with_prob.describe()
    sub_with_prob.to_csv("submission_with_prob.csv", index=False)


if __name__ == "__main__":
    USE_PROBABILITY = False
    submission_writer("./model/InceptionResNetV2-0515-5-fold", USE_PROBABILITY)
