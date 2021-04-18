"""
Model Analysis on Extra Dataset:
- Download Dataset in plant-pathology-2020-fgvc7
- Move all 'Test_' images to ./plant-pathology-2020-fgvc7/test
- Move all 'Train_' images to ./plant-pathology-2020-fgvc7/train
Extra Dataset has { 'healthy' 'multiple_diseases' 'rust' 'scab' }
    - T: Predict Label = True Extra Data Label
    - F: Predict Label != True Extra Data Label
    - M: When True Extra Data Label is 'multiple_diseases' and our model predict label is in not 'healthy'
    - P: Predict Label is a part of True Extra Data Label
Use the submission.csv to calculate model generalization
"""

import pandas as pd
import numpy as np

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
predict_data_table = pd.read_csv("./submission.csv", encoding='utf-8')
label_true = list(extra_data_table["labels"])
print(extra_data_table["labels"].value_counts())
label_pre = []
flag = []
for name in image_name:
    label_p = predict_data_table.loc[predict_data_table["image"] == name]["labels"].values[0]
    label_pre.append(label_p)
    label_t = extra_data_table.loc[extra_data_table["image"] == name]["labels"].values[0]
    if label_p == label_t:
        # Mean True
        flag.append("T")
    elif label_t == "multiple_diseases":
        # Mean Maybe True
        if label_p != "healthy":
            flag.append("M")
        else:
            # Mean False
            flag.append("F")
    elif label_t != label_p:
        if label_t in label_p.split(' '):
            # Mean Part True
            flag.append("P")
        else:
            flag.append("F")
merge_data = pd.DataFrame({'image': image_name, 'labels_predict': label_pre, 'label_true': label_true, 'flag': flag})
flag_count = merge_data["flag"].value_counts()
print(flag_count)
print(merge_data.loc[merge_data["flag"] == "F"]["label_true"].value_counts())
merge_data.to_csv("extra_data.csv", index=False)
