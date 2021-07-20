import math
import tensorflow as tf
import efficientnet.tfkeras as efn
import os
import numpy as np
import tensorflow_addons as tfa
import argparse

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Config Inference.")
    parser.add_argument('--test_image', default='./test_images/ad8770db05586b59.jpg', type=str)
    parser.add_argument('--model_path', default='./model', type=str)
    parser.add_argument('--use_tta', default=False, action="store_true")
    parser.add_argument('--random_crop', default=False, action="store_true")
    parser.add_argument('--tta_step', default=4, type=int)
    parser.add_argument('--resize', default=600, type=int)
    parser.add_argument('--crop', default=512, type=int)
    parser.add_argument('--use_probability', default=False, action="store_true")
    parser.add_argument('--prob_vector', type=float, nargs='+', default=[0.5, 0.5, 0.5, 0.5, 0.5])
    return parser


parser = get_parser()
args = parser.parse_args()

cfg = {
    'data_params': {
        'img_shape': (args.crop, args.crop),
        'over_bound_img_shape': (args.resize, args.resize),
    },
    'model_params': {
        'standardization': False,
        'random_resize': args.random_crop
    }
}

classes = np.array([
    'scab',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew'])

K = 5
mean = [124.23002308, 159.76066492, 104.05509866]
std = [47.84116963, 41.94039282, 49.85093766]
AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_N = 5
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

HEIGHT_T, WIDTH_T = cfg['data_params']['img_shape']
HEIGHT_LARGE, WIDTH_LARGE = cfg['data_params']['over_bound_img_shape']
USE_TTA = args.use_tta
if USE_TTA:
    TTA_STEP = args.tta_step
else:
    TTA_STEP = 1


def _preprocess_image_test_function(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.cast(image, tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if cfg['model_params']['random_resize'] & USE_TTA:
        image = tf.image.resize(image, [HEIGHT_LARGE, WIDTH_LARGE])
    else:
        image = tf.image.resize(image, [HEIGHT_T, WIDTH_T])
    if cfg['model_params']['standardization']:
        i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
        i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
        i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
        image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)],
                          axis=2)
    if USE_TTA:
        image = tf.image.random_contrast(image, lower=1.0, upper=1.3)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        angle = tf.random.uniform([], minval=-np.pi / 6, maxval=np.pi / 6)
        image = tfa.image.rotate(image, angle)
        if cfg['model_params']['random_resize']:
            image = tf.image.random_crop(image, [HEIGHT_T, WIDTH_T, 3])
    return image, os.path.basename(path)


def create_test_model():
    backbone = efn.EfficientNetB7(
        include_top=False,
        input_shape=(HEIGHT_T, WIDTH_T, 3),
        weights=None,
        pooling='avg'
    )
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    return model


def inference(count, model_path, image_path):
    model.load_weights(model_path + "/model_best_%d.h5" % count)
    image, name = _preprocess_image_test_function(image_path)
    pred = model.predict_on_batch(tf.reshape(image, [-1, HEIGHT_T, WIDTH_T, 3]))
    return np.reshape(pred, CLASS_N)


if __name__ == "__main__":
    USE_PROBABILITY = args.use_probability
    probability = args.prob_vector
    print(probability)
    model = create_test_model()
    model.summary()
    preds = []
    for i in tqdm(range(K * TTA_STEP)):
        pred = inference(math.floor(i / TTA_STEP), args.model_path, args.test_image)
        preds.append(pred)
    preds = np.sum(np.array(preds) / (K * TTA_STEP), axis=0)
    if USE_PROBABILITY:
        probs = preds > probability
    else:
        probs = np.around(preds)
    probs = probs.astype('bool')
    probs = np.reshape(probs, CLASS_N)
    label = ' '.join(classes[probs])
    if label == '':
        label = 'healthy'
    print(label)
    print(preds)
