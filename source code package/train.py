import random
import os
import numpy as np
import tensorflow as tf
import math
import pandas as pd
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
from tqdm import tqdm
import efficientnet.tfkeras as efn
import tensorflow_probability as tfp
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tensorflow.keras import backend as K
import argparse


k_fold = 5
mean = [124.23002308, 159.76066492, 104.05509866]
std = [47.84116963, 41.94039282, 49.85093766]
CLASS_N = 5


def get_parser():
    parser = argparse.ArgumentParser(description="Config Train.")
    parser.add_argument('--train_images', default='./train_images/', type=str)
    parser.add_argument('--label', default='./train_without_rep.csv', type=str)
    parser.add_argument('--mix_up', default=False, action="store_true")
    parser.add_argument('--random_resize', default=False, action="store_true")
    parser.add_argument('--label_smooth', default=False, action="store_true")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--iteration', default=128, type=int)
    parser.add_argument('--learning_rate_max', default=1e-3, type=float)
    parser.add_argument('--learning_rate_min', default=1e-6, type=float)
    parser.add_argument('--cycle', default=10, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--pseudo_labels', default=False, action="store_true")
    parser.add_argument('--use_sgd', default=False, action="store_true")
    return parser


parser = get_parser()
args = parser.parse_args()

cfg = {
    'data_params': {
        'img_shape': (512, 512),
        'over_bound_img_shape': (600, 600),
    },
    'model_params': {
        'batchsize_per_gpu': args.batch_size,
        'iteration_per_epoch': args.iteration,
        'epoch': args.epoch,
        'mix-up': args.mix_up,
        'standardization': False,
        'random_resize': args.random_resize,
        'label_smooth': args.label_smooth
    }
}

classes = np.array([
    'scab',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew'])

ls = 0.10

WIDTH = cfg['data_params']['img_shape'][0]
HEIGHT = cfg['data_params']['img_shape'][1]
HEIGHT_LARGE = cfg['data_params']['over_bound_img_shape'][0]
WIDTH_LARGE = cfg['data_params']['over_bound_img_shape'][1]
SEED = 2021
USE_SGD = args.use_sgd
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_ROOT = args.train_images
if cfg['model_params']['random_resize']:
    TFRCORDS_ROOT = "./tfrecords_600/*.tfrecords"
else:
    TFRCORDS_ROOT = "./tfrecords_500/*.tfrecords"
if args.pseudo_labels:
    if cfg['model_params']['random_resize']:
        tfrecs_extra = tf.io.gfile.glob("./tfrecords_600_extra/*.tfrecords")
    else:
        tfrecs_extra = tf.io.gfile.glob("./tfrecords_512_extra/*.tfrecords")
train_data = pd.read_csv("train_without_rep.csv", encoding='utf-8')
data_count = train_data["labels"].value_counts()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)
tfrecs = tf.io.gfile.glob(TFRCORDS_ROOT)
raw_image_dataset = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)

image_feature_description = {
    'name': tf.io.FixedLenFeature([], tf.string),
    'data': tf.io.FixedLenFeature([], tf.string),
    'labels': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'label_name': tf.io.FixedLenFeature([], tf.string)
}


def _parse_image_function(single_photo):
    return tf.io.parse_single_example(single_photo, image_feature_description)


def _preprocess_image_function(single_photo):
    image = tf.io.decode_raw(single_photo['data'], tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if cfg['model_params']['random_resize']:
        image = tf.reshape(image, [HEIGHT_LARGE, WIDTH_LARGE, 3])
    else:
        image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    if cfg['model_params']['standardization']:
        i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
        i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
        i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
        image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)],
                          axis=2)
    image = tf.image.random_jpeg_quality(image, 80, 100)
    gau = tf.keras.layers.GaussianNoise(0.3)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.cond(tf.random.uniform([]) < 0.5,
                    lambda: tf.image.random_saturation(image, lower=0.7, upper=1.3),
                    lambda: tf.image.random_hue(image, max_delta=0.3))
    image = tf.image.random_brightness(image, 0.3)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.squeeze(image, axis=0)
    angle = tf.random.uniform([], minval=-math.pi / 6, maxval=math.pi / 6)
    image = tfa.image.rotate(image, angle)
    image = tf.expand_dims(image, axis=0)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.squeeze(image, axis=0)
    if cfg['model_params']['random_resize']:
        image = tf.image.random_crop(image, [HEIGHT, WIDTH, 3])
    single_photo['data'] = image
    return single_photo


def _preprocess_image_val_function(single_photo):
    image = tf.io.decode_raw(single_photo['data'], tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if cfg['model_params']['random_resize']:
        image = tf.reshape(image, [HEIGHT_LARGE, WIDTH_LARGE, 3])
        image = tf.image.resize(images=image, size=[HEIGHT, WIDTH])
    else:
        image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    if cfg['model_params']['standardization']:
        i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
        i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
        i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
        image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)],
                          axis=2)
    single_photo['data'] = image
    return single_photo


def create_idx_filter(indice):
    def _filt(i, single_photo):
        return tf.reduce_any(indice == i)

    return _filt


def _remove_idx(i, single_photo):
    return single_photo


def _mixup(data, targ):
    # 打乱batch顺序
    indice = tf.range(len(data))
    indice = tf.random.shuffle(indice)
    sinp = tf.gather(data, indice, axis=0)
    starg = tf.gather(targ, indice, axis=0)
    # 生成beta分布
    alpha = 0.2
    t = tfp.distributions.Beta(alpha, alpha).sample([len(data)])
    tx = tf.reshape(t, [-1, 1, 1, 1])
    ty = tf.reshape(t, [-1, 1])
    x = data * tx + sinp * (1 - tx)
    zeros = tf.zeros([cfg['model_params']['batchsize_per_gpu'], HEIGHT, WIDTH, 3])
    x = tf.math.add(x, zeros)
    targ_zeros = tf.zeros([CLASS_N], tf.float32)
    y = targ * ty + starg * (1 - ty)
    y = tf.math.add(y, targ_zeros)
    return x, y


def _create_annot(single_photo):
    targ = tf.io.decode_raw(single_photo['labels'], tf.float32)
    zeros = tf.zeros([CLASS_N], tf.float32)
    targ = tf.math.add(targ, zeros)
    return single_photo['data'], targ


def _create_annot_val(single_photo):
    return single_photo['data'], single_photo['name']


indices = []
name = []
label = []
preprocess_dataset = (raw_image_dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
                      .prefetch(AUTOTUNE)
                      .enumerate())

for i, sample in tqdm(preprocess_dataset):
    indices.append(i.numpy())
    mul_label = list(tf.io.decode_raw(sample['labels'], tf.float32).numpy())
    label.append(mul_label)
    name.append(sample['name'].numpy().decode())

table = pd.DataFrame({'indices': indices, 'name': name, 'label': label})
skf = MultilabelStratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
X = np.array(table.index)
Y = np.array(list(table.label.values), dtype=np.uint8).reshape(len(train_data), CLASS_N)
splits = list(skf.split(X, Y))
print("DataSet Split Successful.")
print("origin: ", np.sum(np.array(list(table["label"].values), dtype=np.uint8), axis=0))
for j in range(5):
    print("Train Fold", j, ":", np.sum(np.array(list(table["label"][splits[j][0]].values), dtype=np.uint8), axis=0))
    print("Val Fold", j, ":", np.sum(np.array(list(table["label"][splits[j][1]].values), dtype=np.uint8), axis=0))
    with open("./k-fold_" + str(j) + ".txt", 'w') as writer:
        writer.write("Train:\n")
        indic_str = "\n".join([str(l) for l in list(splits[j][0])])
        writer.write(indic_str)
        writer.write("\n")
        writer.write("Val:\n")
        indic_str = "\n".join([str(l) for l in list(splits[j][1])])
        writer.write(indic_str)
    writer.close()


def create_train_dataset(batchsize, train_idx):
    global preprocess_dataset
    if args.pseudo_labels:
        extra_dataset = (tf.data.TFRecordDataset(tfrecs_extra, num_parallel_reads=AUTOTUNE)
                         .map(_parse_image_function, num_parallel_calls=AUTOTUNE)
                         .cache())
    parsed_train = (preprocess_dataset
                    .filter(create_idx_filter(train_idx))
                    .map(_remove_idx))
    if args.pseudo_labels:
        dataset = (parsed_train.concatenate(extra_dataset)
                   .cache()
                   .shuffle(len(train_idx) + 3642)
                   .repeat()
                   .map(_preprocess_image_function, num_parallel_calls=AUTOTUNE)
                   .map(_create_annot, num_parallel_calls=AUTOTUNE)
                   .batch(batchsize))
    else:
        dataset = (parsed_train
                   .cache()
                   .shuffle(len(train_idx))
                   .repeat()
                   .map(_preprocess_image_function, num_parallel_calls=AUTOTUNE)
                   .map(_create_annot, num_parallel_calls=AUTOTUNE)
                   .batch(batchsize))
    if cfg['model_params']['mix-up']:
        dataset = (dataset.map(_mixup, num_parallel_calls=AUTOTUNE)
                   .prefetch(AUTOTUNE))
    else:
        dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def create_val_dataset(batchsize, val_idx):
    global preprocess_dataset
    parsed_val = (preprocess_dataset
                  .filter(create_idx_filter(val_idx))
                  .map(_remove_idx))
    dataset = (parsed_val
               .map(_preprocess_image_val_function, num_parallel_calls=AUTOTUNE)
               .map(_create_annot, num_parallel_calls=AUTOTUNE)
               .batch(batchsize)
               .cache())
    return dataset


def create_val_extra_dataset(batchsize, val_idx):
    global preprocess_dataset
    parsed_val = (preprocess_dataset
                  .filter(create_idx_filter(val_idx))
                  .map(_remove_idx))
    dataset = (parsed_val
               .map(_preprocess_image_val_function, num_parallel_calls=AUTOTUNE)
               .map(_create_annot_val, num_parallel_calls=AUTOTUNE)
               .batch(batchsize)
               .cache())
    return dataset


def f1_score_ours(y_true, y_pred):
    y_true_addon = tf.cast(~(K.sum(y_true, axis=1) > 0), tf.float32)
    y_true_addon = tf.reshape(y_true_addon, [len(y_true_addon), -1])
    y_true = tf.concat([y_true, y_true_addon], 1)
    y_pred_addon = 1 - K.max(y_pred, axis=1)
    y_pred_addon = tf.reshape(y_pred_addon, [len(y_pred_addon), -1])
    y_pred = tf.concat([y_pred, y_pred_addon], 1)
    y_pred = tf.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=1)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=1)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=1)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=1)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    y_true_addon = tf.cast(~(K.sum(y_true, axis=1) > 0), tf.float32)
    y_true_addon = tf.reshape(y_true_addon, [len(y_true_addon), -1])
    y_true = tf.concat([y_true, y_true_addon], 1)
    if cfg['model_params']['label_smooth']:
        y_true = (1 - ls) * y_true + ls / CLASS_N
    y_pred_addon = 1 - K.max(y_pred, axis=1)
    y_pred_addon = tf.reshape(y_pred_addon, [len(y_pred_addon), -1])
    y_pred = tf.concat([y_pred, y_pred_addon], 1)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=1)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=1)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=1)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=1)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


class CosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            global_steps,
            learning_rate_max,
            learning_rate_min,
            cycle):
        super().__init__()
        self.global_steps = tf.cast(global_steps, dtype=tf.float32)
        self.learning_rate_max = tf.cast(learning_rate_max, dtype=tf.float32)
        self.learning_rate_min = tf.cast(learning_rate_min, dtype=tf.float32)
        self.cycle = tf.cast(cycle, dtype=tf.int32)
        self.learning_rate = tf.Variable(0., tf.float32)

    def __call__(self, step):
        step_epoch = tf.cast(step, tf.float32) / tf.cast(cfg['model_params']['iteration_per_epoch'], tf.float32)
        step_epoch = tf.cast(step_epoch, tf.int32)
        learning_rate = self.learning_rate_min + 0.5 * (self.learning_rate_max - self.learning_rate_min) * \
                        (1 + tf.math.cos(tf.constant(math.pi, tf.float32) *
                                         (tf.cast(step_epoch % self.cycle, tf.float32) / tf.cast(self.cycle,
                                                                                                 tf.float32))))
        self.learning_rate.assign(learning_rate)
        return learning_rate

    def get_config(self):
        return {
            "global_steps": self.global_steps,
            "learning_rate_max": self.learning_rate_max,
            "learning_rate_min": self.learning_rate_min,
            "cycle": self.cycle
        }

    def return_lr(self):
        return self.learning_rate


class ShowLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.return_lr()
        print("lr:", lr.numpy())


def create_model():
    backbone = efn.EfficientNetB7(
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3),
        weights='noisy-student',
        pooling='avg'
    )
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.he_normal(), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')
    ])
    return model


def plot_history(history, name):
    plt.figure(figsize=(24, 4))
    plt.subplot(1, 4, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("loss")
    plt.subplot(1, 4, 2)
    plt.plot(history.history["f1_score"])
    plt.plot(history.history["val_f1_score"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("Macro F1-Score")
    plt.subplot(1, 4, 3)
    plt.plot(history.history["f1_score_ours"])
    plt.plot(history.history["val_f1_score_ours"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("Sample F1-Score")
    plt.subplot(1, 4, 4)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("metric")
    plt.savefig(name)


model = create_model()


def train(splits, split_id):
    batchsize = cfg['model_params']['batchsize_per_gpu']
    print("batchsize", batchsize)
    if USE_SGD:
        learning_rate = CosineAnnealing(
            global_steps=cfg['model_params']['epoch'] * cfg['model_params']['iteration_per_epoch'],
            learning_rate_max=args.learning_rate_max,
            learning_rate_min=args.learning_rate_min,
            cycle=args.cycle)
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, decay=0.0001)
    else:
        learning_rate = CosineAnnealing(
            global_steps=cfg['model_params']['epoch'] * cfg['model_params']['iteration_per_epoch'],
            learning_rate_max=args.learning_rate_max,
            learning_rate_min=args.learning_rate_min,
            cycle=args.cycle)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)

    model = create_model()
    model.compile(optimizer=optimizer,
                  loss=f1_loss,
                  metrics=['accuracy',
                           tfa.metrics.F1Score(num_classes=CLASS_N, threshold=0.5, average='macro'),
                           f1_score_ours])
    idx_train_tf = tf.cast(tf.constant(splits[split_id][0]), tf.int64)
    idx_val_tf = tf.cast(tf.constant(splits[split_id][1]), tf.int64)
    dataset = create_train_dataset(batchsize, idx_train_tf)
    vdataset = create_val_dataset(batchsize, idx_val_tf)
    history = model.fit(dataset,
                        batch_size=cfg['model_params']['batchsize_per_gpu'],
                        steps_per_epoch=cfg['model_params']['iteration_per_epoch'],
                        epochs=cfg['model_params']['epoch'],
                        validation_data=vdataset,
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath='./model_best_%d.h5' % split_id,
                                save_weights_only=True,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True),
                            ShowLR()
                        ])
    plot_history(history, 'history_%d.png' % split_id)


if __name__ == "__main__":
    if not os.path.exists('./tfrecords_600'):
        print("tfrecords_600 not generated.")
        exit(-1)
    if not os.path.exists('./tfrecords_512'):
        print("tfrecords_512 not generated.")
        exit(-1)
    for i in range(k_fold):
        train(splits, i)
