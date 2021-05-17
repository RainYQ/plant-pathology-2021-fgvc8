"""
    Transform "submission_with_prob.csv" to "submission.csv"
    - Can use probability
"""


import pandas as pd
import numpy as np
from config import classes

USE_PROBABILITY = True
probability = [0.4, 0.5, 0.6, 0.5, 0.5]
CLASS_N = 5
data = pd.read_csv("./model/EfficientNetB7-0512-5-fold/submission_with_prob.csv")
labels = []
names = []
for index, row in data.iterrows():
    names.append(row[0])
    probs = np.array(row[1:CLASS_N + 1], dtype=np.float32)
    if USE_PROBABILITY:
        prob = probs > probability
    else:
        prob = np.around(probs)
    label = ' '.join(classes[prob])
    if label == '':
        label = 'healthy'
    labels.append(label)
sub = pd.DataFrame({
    'image': names,
    'labels': labels})
sub.to_csv('submission.csv', index=False)
