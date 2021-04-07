"""
Train:
Modify:
    - Line 35-47 cfg
    - Line 366 create_test_model
    - Line 512 model location
    - Line 337 create_model
    - Line 355 lr
    - Line 342 Model struct
    - Line 365 Loss
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
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from tqdm import tqdm
import efficientnet.tfkeras as efn
from PIL import Image
from sklearn.metrics import f1_score
from Preprocess import id2label

# 是否仅测试
test_only = True
# 是否对数据集进行过采样
# !效果非常不好
upsample = False

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# R G B 在测试集上的均值、标准差
# range 0 - 255
mean = [124.23002308, 159.76066492, 104.05509866]
std = [47.84116963, 41.94039282, 49.85093766]

cfg = {
    'data_params': {
        'img_shape': (256, 256),
        'test_img_shape': (512, 512),
        'class_type': 6
    },
    'model_params': {
        'batchsize_per_gpu': 16,
        'iteration_per_epoch': 128,
        'batchsize_in_test': 16,
        'epoch': 100
    }
}

classes = np.array([
    'scab',
    'healthy',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew'])

CLASS_N = cfg['data_params']['class_type']

HEIGHT = cfg['data_params']['img_shape'][0]
WIDTH = cfg['data_params']['img_shape'][1]
HEIGHT_T = cfg['data_params']['test_img_shape'][0]
WIDTH_T = cfg['data_params']['test_img_shape'][1]
SEED = 2021
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_ROOT = "./train_images"
TEST_DATA_ROOT = "./test_images"
train_data = pd.read_csv("./train_without_rep.csv", encoding='utf-8')
data_count = train_data["labels"].value_counts()
prob = []
for k in range(12):
    prob.append(data_count[k] / len(train_data))
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
    # image = tf.image.per_image_standardization(image)
    i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
    i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
    i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
    image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=2)
    return image, name


def _preprocess_image_function(single_photo):
    image = tf.image.convert_image_dtype(single_photo['data'], tf.float32)
    image = tf.image.resize(images=image, size=[HEIGHT, WIDTH])
    i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
    i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
    i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
    # use the all dataset data
    image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=2)
    # image = tf.image.per_image_standardization(image)
    # 高斯噪声的标准差为0.3
    gau = tf.keras.layers.GaussianNoise(0.3)
    # 以50％的概率为图像添加高斯噪声
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # brightness随机调整
    image = tf.image.random_brightness(image, 0.2)
    # random left right flip
    image = tf.image.random_flip_left_right(image)
    # random up down flip
    image = tf.image.random_flip_up_down(image)
    rand_k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=SEED)
    # 以50％的概率随机翻转图像
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: tf.image.rot90(image, k=rand_k), lambda: image)
    single_photo['data'] = image
    return single_photo


def _preprocess_image_val_function(single_photo):
    # image = tf.expand_dims(single_photo['data'], axis=0)
    image = tf.image.convert_image_dtype(single_photo['data'], tf.float32)
    image = tf.image.resize(images=image, size=[HEIGHT, WIDTH])
    # image = tf.image.per_image_standardization(image)
    i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
    i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
    i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
    # use the all dataset data
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
    # targ = tf.one_hot(single_photo["label"], CLASS_N, off_value=0)
    # targ = tf.cast(targ, tf.float32)
    targ = tf.io.decode_raw(single_photo['labels'], tf.float32)
    return single_photo['data'], targ


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
label_index = []
for j in range(12):
    label_index.append([])
for i, sample in tqdm(preprocess_dataset):
    indices.append(i.numpy())
    label.append(sample['label'].numpy())
    label_index[sample['label'].numpy()].append(i.numpy())
    name.append(sample['name'].numpy().decode())

table = pd.DataFrame({'indices': indices, 'name': name, 'label': label})
skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
splits = list(skf.split(table.index, table.label))


# Verify KFold. Pass...
# data_count = table['label'][splits[0][1]].value_counts()
# print(data_count)
# plt.bar([i for i in range(12)], data_count)
# plt.xlabel('label type')
# plt.ylabel('count')
# plt.show()


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
               .batch(batchsize)
               .prefetch(AUTOTUNE))
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


def create_model():
    # backbone = tf.keras.applications.ResNet50(weights="imagenet", include_top=False,
    #                                           input_shape=(HEIGHT, WIDTH, 3), classes=CLASS_N)
    backbone = efn.EfficientNetB0(
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3),
        weights='noisy-student',
        pooling='avg'
    )

    model = tf.keras.Sequential([
        backbone,
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, bias_initializer=tf.keras.initializers.Constant(-2.))])
    # optimizer = tfa.optimizers.RectifiedAdam(lr=1e-4,
    #                                          total_steps=cfg['model_params']['iteration_per_epoch'] *
    #                                                      cfg['model_params']['epoch'],
    #                                          warmup_proportion=0.1,
    #                                          min_lr=1e-6)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # 使用FacalLoss和RectifiedAdam
    model.compile(optimizer=optimizer,
                  loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=CLASS_N, threshold=0.5, average='macro')])
    return model


def create_test_model():
    backbone = efn.EfficientNetB4(
        include_top=False,
        input_shape=(HEIGHT_T, WIDTH_T, 3),
        weights=None,
        pooling='avg'
    )

    model = tf.keras.Sequential([
        backbone,
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, bias_initializer=tf.keras.initializers.Constant(-2.))])
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer,
                  loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=CLASS_N, threshold=0.5, average='macro')])
    return model


def plot_history(history, name):
    plt.figure(figsize=(16, 3))
    plt.subplot(1, 3, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("loss")
    # plt.yscale('log')
    plt.subplot(1, 3, 2)
    plt.plot(history.history["f1_score"])
    plt.plot(history.history["val_f1_score"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("F1-Score")
    plt.subplot(1, 3, 3)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("metric")
    plt.savefig(name)


tdataset = (tf.data.Dataset.from_tensor_slices((test_img_lists, test_img_path_lists))
            .map(_preprocess_image_test_function, num_parallel_calls=AUTOTUNE)
            .batch(cfg['model_params']['batchsize_in_test']).prefetch(AUTOTUNE))
if test_only:
    model = create_test_model()
else:
    model = create_model()


def inference(count, path):
    model.load_weights(path + "/model_best_%d.h5" % count)
    rec_ids = []
    probs = []
    for data, name in tqdm(tdataset):
        pred = model.predict_on_batch(tf.reshape(data, [-1, HEIGHT_T, WIDTH_T, 3]))
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


def submission_writer(path):
    sub_with_prob = sum(
        map(
            lambda j:
            inference(j, path).set_index('name'), range(5)
        )
    ).reset_index()

    labels = []
    names = []
    for index, row in sub_with_prob.iterrows():
        names.append(row[0])
        prob = np.around(np.array(row[1:7], dtype=np.float32))
        prob = prob.astype('bool')
        labels.append(' '.join(classes[prob]))
    sub = pd.DataFrame({
        'image': names,
        'labels': labels})
    sub.to_csv('submission.csv', index=False)
    sub_with_prob.describe()
    sub_with_prob.to_csv("submission_with_prob.csv", index=False)


def train(splits, split_id):
    batchsize = cfg['model_params']['batchsize_per_gpu']
    print("batchsize", batchsize)
    model = create_model()

    idx_train_tf = tf.cast(tf.constant(splits[split_id][0]), tf.int64)
    idx_val_tf = tf.cast(tf.constant(splits[split_id][1]), tf.int64)
    if upsample:
        label_list = []
        for i in range(12):
            label_list.append(tf.cast(tf.constant(list(set(splits[split_id][0]) & set(label_index[i]))), tf.int64))
        dataset_filtered = []
        for j in range(12):
            dataset_filtered.append(create_train_dataset(batchsize, label_list[j]))
        over_dataset = tf.data.experimental.sample_from_datasets(
            dataset_filtered, weights=prob, seed=SEED
        )
    else:
        over_dataset = None
    # 生成训练集和验证集
    dataset = create_train_dataset(batchsize, idx_train_tf)
    vdataset = create_val_dataset(batchsize, idx_val_tf)
    history = model.fit(over_dataset if upsample else dataset,
                        batch_size=cfg['model_params']['batchsize_per_gpu'],
                        steps_per_epoch=cfg['model_params']['iteration_per_epoch'],
                        epochs=cfg['model_params']['epoch'],
                        validation_data=vdataset,
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath='./model/model_best_%d.h5' % split_id,
                                save_weights_only=True,
                                monitor='val_f1_score',
                                mode='max',
                                save_best_only=True),
                            # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score',
                            #                                      mode='max',
                            #                                      patience=5,
                            #                                      factor=0.2,
                            #                                      min_lr=1e-6)
                        ])
    plot_history(history, 'history_%d.png' % split_id)


if not test_only:
    for i in range(5):
        train(splits, i)

submission_writer("./model/EfficientNetB4-0407-Noisy-student-kaggle")
