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
from Preprocess import label2array, label2id, id2label

classes = np.array([
    'scab',
    # 'healthy',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew'])

CLASS_N = 5

train_data = pd.read_csv("./train_without_rep.csv", encoding='utf-8')
data_count = train_data["labels"].value_counts()
print(data_count)
# train_data_extra = pd.read_csv("./train.csv", encoding='utf-8')
# remove = list(set(train_data_extra["image"]).difference(set(train_data["image"])))
result = pd.read_csv("./model/EfficientNetB7-0418-Test03/submission_with_prob_val_0.csv", encoding='utf-8')
# result = result[~result["name"].isin(remove)]
col = list(result.columns.values[1:])
result_arr = np.array(result[col])
result_arr = np.around(result_arr)
names = list(result[result.columns.values[0]])
train_data["labels"] = train_data["labels"].map(label2id)
t_label = []
t_label_name = []
p_label_name = []
flag = []
j = 0
for name in names:
    label = train_data.loc[train_data["image"] == name]["labels"]
    t_label.append(label2array[int(label)])
    t_label_name.append(id2label[int(label)])
    prob = result_arr[j].astype('bool')
    label_pre = ' '.join(classes[prob])
    if label_pre == '':
        label_pre = 'healthy'
    p_label_name.append(label_pre)
    if id2label[int(label)] == label_pre:
        # Mean True
        flag.append("T")
    else:
        flag.append("F")
    j += 1
t_label = np.array(t_label, np.float32)
merge_data = pd.DataFrame({'image': names, 'labels_predict': p_label_name, 'label_true': t_label_name, 'flag':flag})
print(f1_score(t_label, result_arr, average='samples'))
print(multilabel_confusion_matrix(t_label, result_arr))
print(classification_report(t_label, result_arr))
flag_count = merge_data["flag"].value_counts()
print(flag_count)
print(merge_data.loc[merge_data["flag"] == "F"]["label_true"].value_counts())
merge_data.to_csv("./model/EfficientNetB7-0418-Test03/val_data_analysis.csv", index=False)