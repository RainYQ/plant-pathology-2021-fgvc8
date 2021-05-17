"""
    Ensemble (Only Average) All submission_with_prob.csv
    - submission directory: "./model/results"
    - Write "./submission_with_prob.csv" (with probability) "./submission.csv" ( Only labels )
    - threshold: 0.5
"""

import pandas as pd
import os
import numpy as np
from config import classes


def ensemble(DataFrame):
    DataFrame[0] = 1 / len(DataFrame) * DataFrame[0].set_index("name")
    for j in range(len(DataFrame) - 1):
        DataFrame[j + 1] = DataFrame[j + 1].set_index("name")
        DataFrame[0] += 1 / len(DataFrame) * DataFrame[j + 1]
    return DataFrame[0].reset_index()


CLASS_N = 5
workdir = "./model/results"
filenames = os.listdir(workdir)
DataFrame = []
for filename in filenames:
    DataFrame.append(pd.read_csv(os.path.join(workdir, filename)))
Merge_Data = ensemble(DataFrame)
Merge_Data.to_csv("submission_with_prob.csv", index=False)
labels = []
names = []
for index, row in Merge_Data.iterrows():
    names.append(row[0])
    probs = np.array(row[1:CLASS_N + 1], dtype=np.float32)
    prob = np.around(probs)
    prob = prob.astype('bool')
    label = ' '.join(classes[prob])
    if label == '':
        label = 'healthy'
    labels.append(label)
sub = pd.DataFrame({
    'image': names,
    'labels': labels})
sub.to_csv('submission.csv', index=False)
