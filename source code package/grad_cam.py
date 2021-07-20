import tensorflow as tf
import efficientnet.tfkeras as efn
import cv2
from matplotlib import pyplot as plt

CLASS_N = 5


def grad_cam(model, inputs, class_index):
    x = tf.keras.Input((None, None, 3))
    conv_y = model.get_layer(index=0)(x, training=False)
    y = model.get_layer(index=1)(conv_y, training=False)
    grad_model = tf.keras.Model(x, [conv_y, y])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs, training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    cams = []
    for grad, output in zip(grads, conv_outputs):
        weights = tf.reduce_mean(grad, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1).numpy()
        cams.append(cam)
    return cams


def create_test_model():
    backbone = efn.EfficientNetB7(
        include_top=False,
        input_shape=(512, 512, 3),
        weights=None,
        pooling=None
    )
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.he_normal(), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, kernel_initializer=tf.keras.initializers.he_normal(), activation=None),
        tf.keras.layers.Activation('sigmoid')
    ])
    return model


model = create_test_model()
model.load_weights("./model/model_best_0.h5")
img = tf.keras.preprocessing.image.load_img("../train_images/8afa9687aab53506.jpg", target_size=(512, 512))
inpts = tf.keras.preprocessing.image.img_to_array(img) / 255.0
cams = grad_cam(model, tf.expand_dims(inpts, 0), 2)
plt.figure(figsize=(36, 24))
plt.imshow(plt.imread('../train_images/8afa9687aab53506.jpg'), aspect='auto', interpolation='nearest')
plt.imshow(cv2.resize(cams[0][::-1], (4000, 2672)), cmap='magma', aspect='auto', interpolation='nearest',
           alpha=0.5)
plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.show()
