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
from Preprocess import label2array, label2id

train_data = pd.read_csv("./train_without_rep.csv", encoding='utf-8')
train_data_extra = pd.read_csv("./train.csv", encoding='utf-8')
remove = list(set(train_data_extra["image"]).difference(set(train_data["image"])))
result = pd.read_csv("./model/EfficientNetB4-0407-Noisy-student-kaggle/submission_with_prob.csv", encoding='utf-8')
result = result[~result["name"].isin(remove)]
col = list(result.columns.values[1:])
result_arr = np.array(result[col])
result_arr = np.around(result_arr)
names = list(result[result.columns.values[0]])
train_data["labels"] = train_data["labels"].map(label2id)
t_label = []
for name in names:
    label = train_data.loc[train_data["image"] == name]["labels"]
    t_label.append(label2array[int(label)])
t_label = np.array(t_label, np.float32)
print(f1_score(t_label, result_arr, average='samples'))
print(multilabel_confusion_matrix(t_label, result_arr))
print(classification_report(t_label, result_arr))
