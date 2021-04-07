import random
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
import efficientnet.tfkeras as efn
import pandas as pd
import tensorflow_addons as tfa
from GroupNormalization import GroupNormalization

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

cfg = {
    'data_params': {
        'img_shape': (512, 512)
    },
    'model_params': {
        'batchsize_per_gpu': 16,
        'iteration_per_epoch': 128,
        'epoch': 100
    }
}
CLASS_N = 6
WIDTH = cfg['data_params']['img_shape'][0]
HEIGHT = cfg['data_params']['img_shape'][1]
label2id = {
    'scab': 0,
    'healthy': 1,
    'frog_eye_leaf_spot': 2,
    'rust': 3,
    'complex': 4,
    'powdery_mildew': 5
}

classes = np.array([
    'scab',
    'healthy',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew'])

id2label = dict([(value, key) for key, value in label2id.items()])

AUTOTUNE = tf.data.experimental.AUTOTUNE

TEST_DATA_ROOT = "./test_images"

test_img_lists = os.listdir(TEST_DATA_ROOT)
test_img_path_lists = [os.path.join(TEST_DATA_ROOT, name) for name in test_img_lists]


def create_model():
    backbone = efn.EfficientNetB4(
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3),
        weights='noisy-student',
        pooling='avg'
    )

    model = tf.keras.Sequential([
        backbone,
        # tf.keras.layers.GlobalAveragePooling2D(),
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, bias_initializer=tf.keras.initializers.Constant(-2.))])
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer,
                  loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False),
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=CLASS_N, threshold=0.5, average='macro')])
    return model


def _preprocess_image_test_function(name, path):
    image = tf.io.read_file(path, 'rb')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(images=image, size=[HEIGHT, WIDTH])
    image = tf.image.per_image_standardization(image)
    # i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
    # i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
    # i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
    # image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=2)
    return image, name


tdataset = (tf.data.Dataset.from_tensor_slices((test_img_lists, test_img_path_lists))
            .map(_preprocess_image_test_function, num_parallel_calls=AUTOTUNE)
            .batch(64).prefetch(AUTOTUNE))

model = create_model()


def inference(count):
    global model
    model.load_weights("./model/EfficientNetB4-0407-Noisy-student-kaggle/model_best_%d.h5" % count)
    rec_ids = []
    probs = []
    for data, name in tqdm(tdataset):
        pred = model.predict_on_batch(tf.reshape(data, [-1, HEIGHT, WIDTH, 3]))
        prob = tf.sigmoid(pred)
        rec_id_stack = tf.reshape(name, [-1, 1])
        for rec in name.numpy():
            assert len(np.unique(rec)) == 1
        rec_ids.append(rec_id_stack.numpy()[:, 0])
        probs.append(prob.numpy())
    crec_ids = np.concatenate(rec_ids)
    cprobs = np.concatenate(probs)
    sub_with_prob = pd.DataFrame({
        'name': list(map(lambda x: x.decode(), crec_ids.tolist())),
        **{id2label[i]: cprobs[:, i] / 5 for i in range(CLASS_N)}
    })
    sub_with_prob = sub_with_prob.sort_values('name')
    return sub_with_prob


sub_with_prob = sum(
    map(
        lambda j:
        inference(j).set_index('name'), range(5)
    )
).reset_index()

labels = []
names = []
for index, row in sub_with_prob.iterrows():
    names.append(row[0])
    prob = np.around(np.array(row[1:7],dtype=np.float32))
    prob = prob.astype('bool')
    labels.append(' '.join(classes[prob]))
sub = pd.DataFrame({
    'image': names,
    'labels': labels})
sub.to_csv('submission.csv', index=False)
sub_with_prob.describe()
sub_with_prob.to_csv("submission_with_prob.csv", index=False)