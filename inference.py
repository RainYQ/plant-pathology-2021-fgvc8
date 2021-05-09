"""
Inference on k-fold Model:
Modify:
    - Line 23 K
    - Line 25 USE_TTA
    - Line 28 TTA_STEP
    - Line 181 model location
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
K = 1
# TTA(测试时增强)
USE_TTA = False
# TTA增强测试次数
if USE_TTA:
    TTA_STEP = 4
else:
    TTA_STEP = 1
SEED = 2021

mean = [124.23002308, 159.76066492, 104.05509866]
std = [47.84116963, 41.94039282, 49.85093766]

AUTOTUNE = tf.data.experimental.AUTOTUNE

CLASS_N = cfg['data_params']['class_type']


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)

probability = [5704, 4350, 2027, 2124, 1271]
probability = np.array(probability, dtype=np.float32) / sum(probability)
print(probability)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

HEIGHT_T, WIDTH_T= cfg['data_params']['test_img_shape']
TEST_DATA_ROOT = "./plant-pathology-2020-fgvc7/train"
test_img_lists = os.listdir(TEST_DATA_ROOT)
test_img_path_lists = [os.path.join(TEST_DATA_ROOT, name) for name in test_img_lists]

def _preprocess_image_test_function(name, path):
    image = tf.io.read_file(path, 'rb')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(images=image, size=[HEIGHT_T, WIDTH_T])
    if cfg['model_params']['standardization']:
        image = tf.image.per_image_standardization(image)
        i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
        i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
        i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
        image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=2)
    if USE_TTA:
        # 高斯噪声的标准差为 0.3
        gau = tf.keras.layers.GaussianNoise(0.3)
        # 以 50％ 的概率为图像添加高斯噪声
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        # brightness随机调整
        image = tf.image.random_brightness(image, 0.3)
        # random left right flip
        image = tf.image.random_flip_left_right(image)
        # random up down flip
        image = tf.image.random_flip_up_down(image)
        # cutout ~2 patches / image
        # width / height 20
        image = tf.expand_dims(image, axis=0)
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
        image = tf.squeeze(image, axis=0)
        # 随机旋转图片 0 ~ 30°
        angle = tf.random.uniform([], minval=0, maxval=30)
        image = tfa.image.rotate(image, angle)
        image = tf.expand_dims(image, axis=0)
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
        image = tf.squeeze(image, axis=0)
        image = tf.image.random_jpeg_quality(image, 80, 100)
        image = tf.image.random_crop(image, [512, 512, 3])
    return image, name


def create_test_model():
    backbone = efn.EfficientNetB0(
        include_top=False,
        input_shape=(HEIGHT_T, WIDTH_T, 3),
        weights=None,
        pooling='avg'
    )

    model = tf.keras.Sequential([
        backbone,
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.he_normal(), activation='relu'),
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    return model


tdataset = (tf.data.Dataset.from_tensor_slices((test_img_lists, test_img_path_lists))
            .map(_preprocess_image_test_function, num_parallel_calls=AUTOTUNE)
            .batch(cfg['model_params']['batchsize_in_test']).prefetch(AUTOTUNE))


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
    submission_writer("./model", USE_PROBABILITY)
