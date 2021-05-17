"""
    Transform The label csv in plant-pathology-2020-fgvc7 to plant-pathology-2021-fgvc8
    - Read "./plant-pathology-2020-fgvc7/train.csv"
    - Write "./plant-pathology-2020-fgvc7/train_label.csv"
    - columns 'image' 'labels'
"""

import numpy as np
import pandas as pd

from config import classes_extra

extra_train_data = pd.read_csv("./plant-pathology-2020-fgvc7/train.csv", encoding='utf-8')
extra_train_data = extra_train_data.sort_values('image_id')
labels_train = []
name = []
for index in extra_train_data.index:
    name.append(extra_train_data.loc[index].values[0] + '.jpg')
    label = ' '.join(classes_extra[np.array(list(extra_train_data.loc[index].values[1:])).astype('bool')])
    labels_train.append(label)
sub = pd.DataFrame({
    'image': name,
    'labels': labels_train})
sub.to_csv("./plant-pathology-2020-fgvc7/train_label.csv", index=False)
