"""MIT License

Copyright (c) 2019 SICARA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# About 5-6s/picture
# Too Slow
# Todo

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from GroupNormalization import GroupNormalization
import efficientnet.tfkeras as efn
import pandas as pd
import os
from Preprocess import label2id, id2label
from tqdm import tqdm
from config import classes

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

CLASS_N = 5
mean = [124.23002308, 159.76066492, 104.05509866]
std = [47.84116963, 41.94039282, 49.85093766]


def grad_cam(model, inputs, class_index):
    x = tf.keras.Input((None, None, 3))
    conv_y = model.get_layer(index=0)(x, training=False)
    y = model.get_layer(index=1)(conv_y, training=False)
    grad_model = tf.keras.Model(x, [conv_y, y])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs, training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    cams = []
    for grad, output in zip(grads, conv_outputs):
        weights = tf.reduce_mean(grad, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1).numpy()
        cams.append(cam)
    return cams


val_data = pd.read_csv('./train_without_rep.csv', encoding='utf-8')
val_img_lists = list(val_data["image"])
AUTOTUNE = tf.data.experimental.AUTOTUNE
VAL_DATA_ROOT = "./train_images"
val_img_path_lists = [os.path.join(VAL_DATA_ROOT, name) for name in val_img_lists]


def _preprocess_image_test_function(name, path):
    image = tf.io.read_file(path, 'rb')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(images=image, size=[512, 512])
    i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
    i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
    i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
    image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=2)
    return image


vdataset = (tf.data.Dataset.from_tensor_slices((val_img_lists, val_img_path_lists))
            .prefetch(AUTOTUNE))


def create_model():
    backbone = efn.EfficientNetB7(
        include_top=False,
        input_shape=(512, 512, 3),
        weights='noisy-student'
    )

    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    return model


def attention(model, pathstr):
    fig, ax = plt.subplots(1, 2, figsize=(32, 8))
    for i, group in tqdm(enumerate(vdataset)):
        name, path = group
        image = tf.io.read_file(path, 'rb')
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(images=image, size=[512, 512])
        image = tf.image.convert_image_dtype(image, tf.uint8)
        inpts = image.numpy()
        true_label = train_data.loc[train_data["image"] == name]["labels"]
        str_name = train_data.loc[train_data["image"] == name]["image"]
        ax[0].set_title(str(str_name.values[0]) + " " + id2label[int(true_label)])
        ax[0].imshow(inpts, aspect='auto', interpolation='nearest')
        outs = _preprocess_image_test_function(name, path)
        pred = model.predict_on_batch(tf.reshape(outs, [-1, 512, 512, 3]))
        prob = np.around(pred.reshape(CLASS_N))
        prob = prob.astype('bool')
        pred_label = ' '.join(classes[prob])
        # 很重要，视为疾病检测模型，没有检测到疾病时视为健康
        if pred_label == '':
            pred_label = 'healthy'
        cams = grad_cam(model, tf.expand_dims(outs, 0), int(true_label))
        ax[1].set_title("grad-cam " + str(str_name.values[0]) + " " + pred_label)
        ax[1].imshow(inpts, aspect='auto', interpolation='nearest')
        ax[1].imshow(cv2.resize(cams[0][::-1], (512, 512)), cmap='magma', aspect='auto', interpolation='nearest',
                     alpha=0.5)
        plt.savefig(pathstr + os.path.splitext(str(str_name.values[0]))[0] + '.png')
        plt.cla()


model = create_model()
train_data = pd.read_csv("./train_without_rep.csv", encoding='utf-8')
train_data["labels"] = train_data["labels"].map(label2id)
# print('Before Train:')
# attention(model, './model_attention_before/')
model.load_weights("./model/EfficientNetB7-0418-Noisy-student-Soft_F1_Loss_Long_Epochs/model_best_0.h5")
print('After Train:')
attention(model, './model_attention_after/')
