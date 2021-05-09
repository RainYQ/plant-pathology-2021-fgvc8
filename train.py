"""
Train:
-- Use EfficientNet-B0 on local machine
-- Use Soft_F1_Loss
-- Use tfa.metrics.F1Score(num_classes=CLASS_N, threshold=0.5, average='macro')
-- Use Adam
-- 0418 Write Model Predict Result over Val_dataset
Modify:
    - Line 33 k-fold number
"""

import random
import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import threading
import tensorflow_addons as tfa
from GroupNormalization import GroupNormalization
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tqdm import tqdm
import efficientnet.tfkeras as efn
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
from config import cfg, classes

USE_PROBABILITY = False
# k-fold number
k_fold = 1

probability = [5704, 4350, 2027, 2124, 1271]
probability = np.array(probability, dtype=np.float32) / sum(probability)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# R G B 在测试集上的均值、标准差
# range 0 - 255
mean = [124.23002308, 159.76066492, 104.05509866]
std = [47.84116963, 41.94039282, 49.85093766]

CLASS_N = cfg['data_params']['class_type']

HEIGHT = cfg['data_params']['img_shape'][0]
WIDTH = cfg['data_params']['img_shape'][1]
HEIGHT_LARGE = cfg['data_params']['over_bound_img_shape'][0]
WIDTH_LARGE = cfg['data_params']['over_bound_img_shape'][1]
HEIGHT_T = cfg['data_params']['test_img_shape'][0]
WIDTH_T = cfg['data_params']['test_img_shape'][1]
SEED = 2021
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_ROOT = "./train_images"
TEST_DATA_ROOT = "./plant-pathology-2020-fgvc7/train"
train_data = pd.read_csv("./train_without_rep.csv", encoding='utf-8')
data_count = train_data["labels"].value_counts()
train_img_lists = os.listdir(TRAIN_DATA_ROOT)
test_img_lists = os.listdir(TEST_DATA_ROOT)
test_img_path_lists = [os.path.join(TEST_DATA_ROOT, name) for name in test_img_lists]


def get_train_set_norm(imglist):
    result_mean_R = []
    result_mean_G = []
    result_mean_B = []
    result_std_R = []
    result_std_G = []
    result_std_B = []

    for i in range(math.ceil(len(train_img_lists) / 128)):
        result_mean_R.append([])
        result_mean_G.append([])
        result_mean_B.append([])
        result_std_R.append([])
        result_std_G.append([])
        result_std_B.append([])
    for img in imglist:
        image = Image.open(os.path.join(TRAIN_DATA_ROOT, img))
        image = image.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        r, g, b = image.split()
        mean_r = np.mean(np.asarray(r).flatten())
        mean_g = np.mean(np.asarray(g).flatten())
        mean_b = np.mean(np.asarray(b).flatten())
        std_r = np.std(np.asarray(r).flatten())
        std_g = np.std(np.asarray(g).flatten())
        std_b = np.std(np.asarray(b).flatten())
        file_name, _ = os.path.splitext(img)
        with open("./mean_std_information/" + file_name + ".txt", 'w') as writer:
            writer.write(str(mean_r) + '\n')
            writer.write(str(mean_g) + '\n')
            writer.write(str(mean_b) + '\n')
            writer.write(str(std_r) + '\n')
            writer.write(str(std_g) + '\n')
            writer.write(str(std_b) + '\n')


# 每次更新WIDTH HEIGHT的时候需要重新运行该函数
# Enable Multi-Thread
def meann_std_generator():
    thread_list = []
    for n in range(math.ceil(len(train_img_lists) / 128)):
        t1 = threading.Thread(target=get_train_set_norm, args=(
            train_img_lists[n * 128:min((n + 1) * 128, len(train_img_lists))],))
        thread_list.append(t1)
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()


# 已知各组标准差，求总体标准差
# S(All) = sqrt(((S(1)^2+S(2)^2+...+S(n)^2)) +
# ((total_mean - group_mean[1])^2+(total_mean - group_mean[2])^2+...+(total_mean - group_mean[n])^2)
# / len(train_img_lists))


def read_mean_std():
    file_list = os.listdir(r"./mean_std_information")
    data = [[], [], [], [], [], []]
    for file in file_list:
        with open('./mean_std_information/' + file) as reader:
            for i in range(6):
                line = reader.readline()
                data[i].append(float(line))
    return data


# data = read_mean_std()


def cal_mean_std(data):
    mean = np.mean(np.array(data[0:3]), axis=1)
    std = np.sqrt(
        (np.sum(np.array(data[3:6]) ** 2, axis=1) + np.array([np.sum((data[0] - mean[0]) ** 2),
                                                              (np.sum((data[1] - mean[1]) ** 2)),
                                                              np.sum((data[2] - mean[2]) ** 2)])) / len(
            train_img_lists))
    return mean, std


# mean, std = cal_mean_std(data)
# print(mean)
# print(std)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)
tfrecs = tf.io.gfile.glob("./train_tfrecords/*.tfrecords")
raw_image_dataset = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)

# Create a dictionary describing the features.
image_feature_description = {
    'name': tf.io.FixedLenFeature([], tf.string),
    'data': tf.io.FixedLenFeature([], tf.string),
    'labels': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'label_name': tf.io.FixedLenFeature([], tf.string)
}


def _parse_image_function(single_photo):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(single_photo, image_feature_description)


def _decode_image_function(single_photo):
    single_photo['data'] = tf.image.decode_jpeg(single_photo['data'], channels=3)
    return single_photo


def _preprocess_image_test_function(name, path):
    image = tf.io.read_file(path, 'rb')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(images=image, size=[HEIGHT_T, WIDTH_T])
    if cfg['model_params']['standardization']:
        i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
        i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
        i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
        image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=2)
    return image, name


def _preprocess_image_function(single_photo):
    image = tf.image.convert_image_dtype(single_photo['data'], tf.float32)
    if cfg['model_params']['random_resize']:
        tf.image.resize(images=image, size=[HEIGHT_LARGE, WIDTH_LARGE])
    else:
        image = tf.image.resize(images=image, size=[HEIGHT, WIDTH])
    if cfg['model_params']['standardization']:
        i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
        i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
        i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
        image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=2)
    image = tf.image.random_jpeg_quality(image, 80, 100)
    # 高斯噪声的标准差为 0.3
    gau = tf.keras.layers.GaussianNoise(0.3)
    # 以 50％ 的概率为图像添加高斯噪声
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    # brightness随机调整
    image = tf.image.random_brightness(image, 0.3)
    # random left right flip
    image = tf.image.random_flip_left_right(image)
    # random up down flip
    image = tf.image.random_flip_up_down(image)
    # cutout ~2 patches / image
    # width / height 20
    image = tf.expand_dims(image, axis=0)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tfa.image.random_cutout(image, [20, 20]), lambda: image)
    image = tf.squeeze(image, axis=0)
    # 随机旋转图片 -30° ~ 30°
    angle = tf.random.uniform([], minval=-30, maxval=30)
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
    image = tf.image.convert_image_dtype(single_photo['data'], tf.float32)
    image = tf.image.resize(images=image, size=[HEIGHT, WIDTH])
    if cfg['model_params']['standardization']:
        i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
        i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
        i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
        image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=2)
    single_photo['data'] = image
    return single_photo


def create_idx_filter(indice):
    def _filt(i, single_photo):
        return tf.reduce_any(indice == i)

    return _filt


def _remove_idx(i, single_photo):
    return single_photo


def _create_annot(single_photo):
    targ = tf.io.decode_raw(single_photo['labels'], tf.float32)
    return single_photo['data'], targ


def _create_annot_val(single_photo):
    return single_photo['data'], single_photo['name']


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
    y = targ * ty + starg * (1 - ty)
    return x, y


# parsed_image_dataset = (raw_image_dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
#                         .map(_decode_image_function, num_parallel_calls=AUTOTUNE)
#                         .map(_preprocess_image_function, num_parallel_calls=AUTOTUNE)
#                         .map(_create_annot, num_parallel_calls=AUTOTUNE))
# for image_features in parsed_image_dataset:
#     # image = image_features['data'].numpy()
#     # image = Image.fromarray(np.uint8(image[0] * 255.0)).convert('RGB')
#     # image.show()
#     print(image_features[0]['name'].numpy().decode())
#     print(image_features[1].numpy())
#     # filename, extension = os.path.splitext(image_features['name'].numpy().decode())
#     # image.save("./preprocess/" + filename + ".png")

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
for j in range(5):
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
    parsed_train = (preprocess_dataset
                    .filter(create_idx_filter(train_idx))
                    .map(_remove_idx))
    dataset = (parsed_train.cache()
               .shuffle(len(train_idx))
               .repeat()
               .map(_decode_image_function, num_parallel_calls=AUTOTUNE)
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
               .map(_decode_image_function, num_parallel_calls=AUTOTUNE)
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
               .map(_decode_image_function, num_parallel_calls=AUTOTUNE)
               .map(_preprocess_image_val_function, num_parallel_calls=AUTOTUNE)
               .map(_create_annot_val, num_parallel_calls=AUTOTUNE)
               .batch(batchsize)
               .cache())
    return dataset


def cal_f1_score(y_true, y_pred):
    y_pred = np.around(y_pred)
    y_true = np.around(y_true)
    return f1_score(y_true, y_pred, average='samples', zero_division=0)


# Use sklearn verify metrics
def f1_score_sk(y_true, y_pred):
    # 将 ‘healthy' 补充到 metrics 函数中
    y_true_addon = tf.cast(~(K.sum(y_true, axis=1) > 0), tf.float32)
    y_true_addon = tf.reshape(y_true_addon, [len(y_true_addon), -1])
    y_true = tf.concat([y_true, y_true_addon], 1)
    y_pred_addon = 1 - K.max(y_pred, axis=1)
    y_pred_addon = tf.reshape(y_pred_addon, [len(y_pred_addon), -1])
    y_pred = tf.concat([y_pred, y_pred_addon], 1)
    return tf.py_function(cal_f1_score, [y_true, y_pred], tf.float32)


# Our loss (soft sample-wise f1 loss)
def f1_loss(y_true, y_pred):
    # axis=0 时 计算出的f1为'macro'
    # axis=1 时 计算出的f1为'samples'
    # 将 ‘healthy' 补充到 loss 函数中
    y_true_addon = tf.cast(~(K.sum(y_true, axis=1) > 0), tf.float32)
    y_true_addon = tf.reshape(y_true_addon, [len(y_true_addon), -1])
    y_true = tf.concat([y_true, y_true_addon], 1)
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


# Our metrics (equals to sklearn.metrics.f1_score('samples'))
def f1_score_ours(y_true, y_pred):
    # axis=0 时 计算出的f1为'macro'
    # axis=1 时 计算出的f1为'samples'
    # 将 ‘healthy' 补充到 metrics 函数中
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
        self.cycle = tf.cast(cycle, dtype=tf.float32)
        self.learning_rate = tf.Variable(0., tf.float32)

    def __call__(self, step):
        learning_rate = self.learning_rate_min + 0.5 * (self.learning_rate_max - self.learning_rate_min) * \
                        (1 + tf.math.cos(tf.constant(math.pi, tf.float32) *
                                         (tf.cast(step, tf.float32) / self.cycle)))
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
    backbone = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3),
        classes=CLASS_N,
        pooling='avg'
    )
    # backbone = efn.EfficientNetB0(
    #     include_top=False,
    #     input_shape=(HEIGHT, WIDTH, 3),
    #     weights='noisy-student',
    #     pooling='avg'
    # )

    model = tf.keras.Sequential([
        backbone,
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    learning_rate = CosineAnnealing(
        global_steps=cfg['model_params']['epoch'] * cfg['model_params']['iteration_per_epoch'],
        learning_rate_max=1e-3,
        learning_rate_min=1e-6,
        cycle=5 * cfg['model_params']['iteration_per_epoch'])
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
    model.compile(optimizer=optimizer,
                  # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False),
                  loss=f1_loss,
                  metrics=['accuracy',
                           tfa.metrics.F1Score(num_classes=CLASS_N, threshold=0.5, average='macro'),
                           f1_score_sk,
                           f1_score_ours
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


# Run Inference On Val Dataset.
# Save as "./submission_val_i.csv"&"./submission_with_prob_val_i.csv"
def inference(count, path, USE_PROBABILITY):
    idx_val_tf = tf.cast(tf.constant(splits[count][1]), tf.int64)
    vdataset = create_val_extra_dataset(cfg['model_params']['batchsize_per_gpu'], idx_val_tf)
    model.load_weights(path + "/model_best_%d.h5" % count)
    rec_ids = []
    probs = []
    for data, name in tqdm(vdataset):
        pred = model.predict_on_batch(tf.reshape(data, [-1, HEIGHT_T, WIDTH_T, 3]))
        rec_id_stack = tf.reshape(name, [-1, 1])
        for rec in name.numpy():
            assert len(np.unique(rec)) == 1
        rec_ids.append(rec_id_stack.numpy()[:, 0])
        probs.append(pred)
    crec_ids = np.concatenate(rec_ids)
    cprobs = np.concatenate(probs)
    sub_with_prob = pd.DataFrame({
        'name': list(map(lambda x: x.decode(), crec_ids.tolist())),
        **{classes[i]: cprobs[:, i] / k_fold for i in range(CLASS_N)}
    })
    sub_with_prob = sub_with_prob.sort_values('name')
    labels = []
    names = []
    for index, row in sub_with_prob.iterrows():
        names.append(row[0])
        probs = np.array(row[1:CLASS_N + 1], dtype=np.float32)
        if USE_PROBABILITY:
            prob = probs > probability
        else:
            prob = np.around(probs)
        prob = prob.astype('bool')
        label = ' '.join(classes[prob])
        # 很重要，视为疾病检测模型，没有检测到疾病时视为健康
        if label == '':
            label = 'healthy'
        labels.append(label)

    sub = pd.DataFrame({
        'image': names,
        'labels': labels})
    sub.to_csv('submission_val_' + str(count) + '.csv', index=False)
    sub_with_prob.describe()
    sub_with_prob.to_csv("submission_with_prob_val_" + str(count) + ".csv", index=False)
    sub_with_prob.set_index('name')
    return sub_with_prob


def train(splits, split_id):
    batchsize = cfg['model_params']['batchsize_per_gpu']
    print("batchsize", batchsize)
    model = create_model()
    idx_train_tf = tf.cast(tf.constant(splits[split_id][0]), tf.int64)
    idx_val_tf = tf.cast(tf.constant(splits[split_id][1]), tf.int64)
    # 生成训练集和验证集
    dataset = create_train_dataset(batchsize, idx_train_tf)
    vdataset = create_val_dataset(batchsize, idx_val_tf)
    history = model.fit(dataset,
                        batch_size=cfg['model_params']['batchsize_per_gpu'],
                        steps_per_epoch=cfg['model_params']['iteration_per_epoch'],
                        epochs=cfg['model_params']['epoch'],
                        validation_data=vdataset,
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath='./model/model_best_%d.h5' % split_id,
                                save_weights_only=True,
                                monitor='val_f1_score_sk',
                                mode='max',
                                save_best_only=True),
                            ShowLR()
                            # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score_sk',
                            #                                      mode='max',
                            #                                      verbose=1,
                            #                                      patience=5,
                            #                                      factor=0.5,
                            #                                      min_lr=1e-6)
                        ])
    plot_history(history, 'history_%d.png' % split_id)


for i in range(k_fold):
    train(splits, i)
    inference(i, "./model", USE_PROBABILITY)
