"""
Find a better threshold
Modify:
    - Line 15 csv location
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from config import id2array, label2id, cfg

CLASS_N = cfg['data_params']['class_type']
CSV_LOCATION = "./model/EfficientNetB7-0508-Mixup/"

train_data = pd.read_csv("./train_without_rep.csv", encoding='utf-8')
result = pd.read_csv(CSV_LOCATION + "submission_with_prob_val_0.csv", encoding='utf-8')
col = list(result.columns.values[1:])
names = list(result[result.columns.values[0]])
train_data["labels"] = train_data["labels"].map(label2id)
t_label_without_healthy = []
preds = []
for name in names:
    label = train_data.loc[train_data["image"] == name]["labels"]
    pred = list(np.array(result.loc[result["name"] == name][col]).reshape(CLASS_N))
    preds.append(pred.copy())
    t_label_without_healthy.append(id2array[int(label)])
t_label_without_healthy = np.array(t_label_without_healthy, np.float32)


def f1_score_our(y_true, y_pred, thresholds):
    y_true_addon = tf.cast(~(K.sum(y_true, axis=1) > 0), tf.float32)
    y_true_addon = tf.reshape(y_true_addon, [len(y_true_addon), -1])
    y_true = tf.concat([y_true, y_true_addon], 1)
    y_true = tf.round(y_true)
    y_pred = tf.cast(y_pred > thresholds, tf.float32)
    # y_pred = tf.round(y_pred)
    y_pred_addon = 1 - K.max(y_pred, axis=1)
    y_pred_addon = tf.reshape(y_pred_addon, [len(y_pred_addon), -1])
    y_pred = tf.concat([y_pred, y_pred_addon], 1)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=1)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=1)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=1)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=1)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


thresholds = np.arange(.01, 1., .01)
scores = []

for threshold in thresholds:
    metric = f1_score_our(t_label_without_healthy, np.array(preds, dtype=np.float32), threshold)
    scores.append(metric.numpy())
print("max score:", max(scores))
print("threshold:", thresholds[np.argmax(scores)])
plt.plot(thresholds, scores)
plt.show()
