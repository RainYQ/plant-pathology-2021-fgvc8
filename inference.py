"""
Inference on 5-fold Model:
Modify:
    - Line 17 model location
"""

import tensorflow as tf
from train import create_test_model, submission_writer

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = create_test_model()

if __name__ == "__main__":
    submission_writer("./model/EfficientNetB4-0407-Noisy-student-kaggle")
