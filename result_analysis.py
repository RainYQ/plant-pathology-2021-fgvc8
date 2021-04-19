"""
Analysis over whole Train Dataset:
    - F1-Score
    - multilabel_confusion_matrix
    - classification_report
Modify:
    - Line 41 csv location
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, multilabel_confusion_matrix, classification_report
from Preprocess import label2id, id2label
import tensorflow as tf
import tensorflow.keras.backend as K

classes = np.array([
    'scab',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew',
    'healthy'])

id2array_with_healthy = {
    0: np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
    1: np.array([0, 0, 0, 0, 0, 1], dtype=np.float32),
    2: np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 1, 0, 0, 0], dtype=np.float32),
    4: np.array([0, 0, 0, 1, 0, 0], dtype=np.float32),
    5: np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),
    6: np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
    7: np.array([1, 1, 0, 1, 0, 0], dtype=np.float32),
    8: np.array([0, 1, 0, 1, 0, 0], dtype=np.float32),
    9: np.array([0, 1, 1, 0, 0, 0], dtype=np.float32),
    10: np.array([0, 0, 1, 1, 0, 0], dtype=np.float32),
    11: np.array([0, 0, 0, 1, 1, 0], dtype=np.float32)
}

id2array = {
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

CLASS_N = 5
CSV_LOCATION = "./model/EfficientNetB7-0418-Test04/"

train_data = pd.read_csv("./train_without_rep.csv", encoding='utf-8')
data_count = train_data["labels"].value_counts()
print(data_count)
# train_data_extra = pd.read_csv("./train.csv", encoding='utf-8')
# remove = list(set(train_data_extra["image"]).difference(set(train_data["image"])))
result = pd.read_csv(CSV_LOCATION + "submission_with_prob_val_0.csv", encoding='utf-8')
# result = result[~result["name"].isin(remove)]
col = list(result.columns.values[1:])
result_arr = np.array(result[col])
# result_arr = np.around(result_arr)
names = list(result[result.columns.values[0]])
train_data["labels"] = train_data["labels"].map(label2id)
t_label = []
t_label_name = []
p_label_name = []
p_label = []
preds = []
flag = []
for name in names:
    label = train_data.loc[train_data["image"] == name]["labels"]
    pred = list(np.array(result.loc[result["name"] == name][col]).reshape(CLASS_N))
    preds.append(pred.copy())
    pred.append(1 - max(pred))
    pred = np.around(np.array(pred))
    p_label.append(pred)
    t_label.append(id2array_with_healthy[int(label)])
    t_label_name.append(id2label[int(label)])
    prob = pred.astype('bool')
    label_pre = ' '.join(classes[prob])
    p_label_name.append(label_pre)
    if id2label[int(label)] == label_pre:
        # Mean True
        flag.append("T")
    else:
        flag.append("F")
t_label = np.array(t_label, np.float32)
p_label = np.array(p_label, np.float32)


# Verify F1-Score.
# Not Pass...
# TODO
def f1_score_our(y_true, y_pred):
    # axis=0 时 计算出的 f1 为 'macro'
    # axis=1 时 计算出的 f1 为 'samples'
    # 将 'healthy' 补充到 metrics 函数中
    y_true_addon = tf.cast(~(K.sum(y_true, axis=1) > 0), tf.float32)
    y_true_addon = tf.reshape(y_true_addon, [len(y_true_addon), -1])
    y_true = tf.concat([y_true, y_true_addon], 1)
    y_true = tf.round(y_true)
    y_pred_addon = 1 - K.max(y_pred, axis=1)
    y_pred_addon = tf.reshape(y_pred_addon, [len(y_pred_addon), -1])
    y_pred = tf.concat([y_pred, y_pred_addon], 1)
    y_pred = tf.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=1)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=1)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=1)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=1)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


merge_data = pd.DataFrame({'image': names, 'labels_predict': p_label_name, 'label_true': t_label_name, 'flag': flag})
print('Our F1-Score', f1_score_our(result_arr, np.array(preds, dtype=np.float32)).numpy())
print('sklearn F1-Score', f1_score(t_label, p_label, average='samples'))
print(multilabel_confusion_matrix(t_label, p_label))
print(classification_report(t_label, p_label))
flag_count = merge_data["flag"].value_counts()
print(flag_count)
print(merge_data.loc[merge_data["flag"] == "F"]["label_true"].value_counts())
merge_data.to_csv(CSV_LOCATION + "val_data_analysis.csv", index=False)
