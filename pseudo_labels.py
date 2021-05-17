"""
    Generate Pseudo Labels for plant-pathology-2020-fgvc7
    0.3 * Our Model Predict + 0.7 * Ground Truth Labels
    -- Train Use train.csv
    -- Test Use submission.csv (0.984)
"""


import os
import pandas as pd
import numpy as np
from config import classes_extra, classes

extra_train_data = pd.read_csv("./plant-pathology-2020-fgvc7/train.csv", encoding='utf-8')
labels_train = []
for index in extra_train_data.index:
    label = ' '.join(classes_extra[np.array(list(extra_train_data.loc[index].values[1:])).astype('bool')])
    labels_train.append(label)
extra_train_data['labels'] = labels_train
extra_train_data[extra_train_data.columns.values[1:5]] = extra_train_data[extra_train_data.columns.values[1:5]].astype(
    'float32')
extra_test_data = pd.read_csv("./plant-pathology-2020-fgvc7/submission.csv", encoding='utf-8')
labels_test = []
for index in extra_test_data.index:
    label = classes_extra[np.argmax(np.array(list(extra_test_data.loc[index].values[1:])))]
    labels_test.append(label)
extra_test_data['labels'] = labels_test
extra_data = pd.concat([extra_train_data, extra_test_data])
predict_data_table = pd.read_csv("./plant-pathology-2020-fgvc7/EfficientNetB7-Predict/submission_with_prob.csv",
                                 encoding='utf-8')
labels_model = []
for index in predict_data_table.index:
    label = ' '.join(
        classes[np.round(np.array(list(predict_data_table.loc[index].values[1:6]), dtype=np.float32)).astype('bool')])
    if label == '':
        label = 'healthy'
    labels_model.append(label)
predict_data_table['labels'] = labels_model
label_pseudos = []
for index in predict_data_table.index:
    # 'scab'
    # 'frog_eye_leaf_spot'
    # 'rust'
    # 'complex'
    # 'powdery_mildew'
    # 'healthy'
    label_predict = np.array(list(predict_data_table.loc[index].values[1:6]), dtype=np.float32)
    healthy_prob = 1 - np.max(label_predict)
    label_predict = np.insert(label_predict, label_predict.shape[0], healthy_prob)
    name = predict_data_table.loc[index].values[0]
    # 'healthy'
    # 'multiple_diseases'
    # 'rust'
    # 'scab'
    label_groud = np.array(extra_data.loc[extra_data["image_id"] == os.path.splitext(name)[0]].iloc[:, 1:5],
                           dtype=np.float32)[0]
    label_groud = np.array([label_groud[3], 0, label_groud[2], label_groud[1], 0, label_groud[0]])
    label_pseudo = 0.3 * label_predict + 0.7 * label_groud
    label_pseudos.append(label_pseudo)
label_pseudos = np.transpose(np.array(label_pseudos, dtype=np.float32))
pseudo_data = pd.DataFrame({'image': predict_data_table["name"], 'scab': label_pseudos[0],
                            'frog_eye_leaf_spot': label_pseudos[1], 'rust': label_pseudos[2],
                            'complex': label_pseudos[3], 'powdery_mildew': label_pseudos[4],
                            'healthy': label_pseudos[5]})
pseudo_data.to_csv("./plant-pathology-2020-fgvc7/pseudo_data.csv", index=False)
