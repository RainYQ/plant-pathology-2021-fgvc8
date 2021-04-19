"""
Analysis over whole Train Dataset:
    - F1-Score
    - multilabel_confusion_matrix
    - classification_report
Modify:
    - Line 18 csv location
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, multilabel_confusion_matrix, classification_report
from Preprocess import label2id, id2label

classes = np.array([
    'scab',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew',
    'healthy'])

id2array = {
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

CLASS_N = 5

train_data = pd.read_csv("./train_without_rep.csv", encoding='utf-8')
data_count = train_data["labels"].value_counts()
print(data_count)
# train_data_extra = pd.read_csv("./train.csv", encoding='utf-8')
# remove = list(set(train_data_extra["image"]).difference(set(train_data["image"])))
result = pd.read_csv("./model/EfficientNetB7-0418-Test04/submission_with_prob_val_0.csv", encoding='utf-8')
# result = result[~result["name"].isin(remove)]
col = list(result.columns.values[1:])
result_arr = np.array(result[col])
result_arr = np.around(result_arr)
names = list(result[result.columns.values[0]])
train_data["labels"] = train_data["labels"].map(label2id)
t_label = []
t_label_name = []
p_label_name = []
p_label = []
flag = []
for name in names:
    label = train_data.loc[train_data["image"] == name]["labels"]
    pred = list(np.array(result.loc[result["name"] == name][col]).reshape(CLASS_N))
    pred.append(1 - max(pred))
    pred = np.around(np.array(pred))
    p_label.append(pred)
    t_label.append(id2array[int(label)])
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
merge_data = pd.DataFrame({'image': names, 'labels_predict': p_label_name, 'label_true': t_label_name, 'flag': flag})
print(f1_score(t_label, p_label, average='samples'))
print(multilabel_confusion_matrix(t_label, p_label))
print(classification_report(t_label, p_label))
flag_count = merge_data["flag"].value_counts()
print(flag_count)
print(merge_data.loc[merge_data["flag"] == "F"]["label_true"].value_counts())
merge_data.to_csv("./model/EfficientNetB7-0418-Test04/val_data_analysis.csv", index=False)
