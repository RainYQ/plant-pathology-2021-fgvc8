# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt

mean = [124.23002308, 159.76066492, 104.05509866]
std = [47.84116963, 41.94039282, 49.85093766]


def _norm_params(mask_size, offset=None):
    tf.assert_equal(
        tf.reduce_any(mask_size % 2 != 0),
        False,
        "mask_size should be divisible by 2",
    )
    if tf.rank(mask_size) == 0:
        mask_size = tf.stack([mask_size, mask_size])
    if offset is not None and tf.rank(offset) == 1:
        offset = tf.expand_dims(offset, 0)
    return mask_size, offset


def cutout(
        images,
        mask_size,
        offset=(0, 0),
        constant_values=0
):
    """Apply [cutout](https://arxiv.org/abs/1708.04552) to images.

    This operation applies a `(mask_height x mask_width)` mask of zeros to
    a location within `images` specified by the offset.
    The pixel values filled in will be of the value `constant_values`.
    The located where the mask will be applied is randomly
    chosen uniformly over the whole images.

    Args:
      images: A tensor of shape `(batch_size, height, width, channels)` (NHWC).
      mask_size: Specifies how big the zero mask that will be generated is that
        is applied to the images. The mask will be of size
        `(mask_height x mask_width)`. Note: mask_size should be divisible by 2.
      offset: A tuple of `(height, width)` or `(batch_size, 2)`
      constant_values: What pixel value to fill in the images in the area that has
        the cutout mask applied to it.
    Returns:
      A `Tensor` of the same shape and dtype as `images`.
    Raises:
      InvalidArgumentError: if `mask_size` can't be divisible by 2.
    """

    images = tf.convert_to_tensor(images)
    mask_size = tf.convert_to_tensor(mask_size)
    offset = tf.convert_to_tensor(offset)

    image_static_shape = images.shape
    image_dynamic_shape = tf.shape(images)
    image_height, image_width, channels = (
        image_dynamic_shape[1],
        image_dynamic_shape[2],
        image_dynamic_shape[3],
    )

    mask_size, offset = _norm_params(mask_size, offset)
    mask_size = mask_size // 2

    cutout_center_heights = offset[:, 0]
    cutout_center_widths = offset[:, 1]

    lower_pads = tf.maximum(0, cutout_center_heights - mask_size[0])
    upper_pads = tf.maximum(0, image_height - cutout_center_heights - mask_size[0])
    left_pads = tf.maximum(0, cutout_center_widths - mask_size[1])
    right_pads = tf.maximum(0, image_width - cutout_center_widths - mask_size[1])

    cutout_shape = tf.transpose(
        [
            image_height - (lower_pads + upper_pads),
            image_width - (left_pads + right_pads),
        ],
        [1, 0],
    )

    def fn(i):
        padding_dims = [
            [lower_pads[i], upper_pads[i]],
            [left_pads[i], right_pads[i]],
        ]
        mask = tf.pad(
            tf.zeros(cutout_shape[i], dtype=tf.bool),
            padding_dims,
            constant_values=True,
        )
        return mask

    mask = tf.map_fn(
        fn,
        tf.range(tf.shape(cutout_shape)[0]),
        fn_output_signature=tf.TensorSpec(
            shape=image_static_shape[1:-1], dtype=tf.bool
        ),
    )
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 1, channels])

    images = tf.where(
        mask,
        images,
        tf.cast(constant_values, dtype=images.dtype),
    )
    images.set_shape(image_static_shape)
    return images


def random_cutout(
        images,
        mask_size,
        constant_values=0,
        seed=None
):
    """Apply [cutout](https://arxiv.org/abs/1708.04552) to images with random offset.

        This operation applies a `(mask_height x mask_width)` mask of zeros to
        a random location within `images`. The pixel values filled in will be of
        the value `constant_values`. The located where the mask will be applied is
        randomly chosen uniformly over the whole images.

        Args:
          images: A tensor of shape `(batch_size, height, width, channels)` (NHWC).
          mask_size: Specifies how big the zero mask that will be generated is that
            is applied to the images. The mask will be of size
            `(mask_height x mask_width)`. Note: mask_size should be divisible by 2.
          constant_values: What pixel value to fill in the images in the area that has
            the cutout mask applied to it.
          seed: A Python integer. Used in combination with `tf.random.set_seed` to
            create a reproducible sequence of tensors across multiple calls.
        Returns:
          A `Tensor` of the same shape and dtype as `images`.
        Raises:
          InvalidArgumentError: if `mask_size` can't be divisible by 2.
        """
    images = tf.convert_to_tensor(images)
    mask_size = tf.convert_to_tensor(mask_size)

    image_dynamic_shape = tf.shape(images)
    batch_size, image_height, image_width = (
        image_dynamic_shape[0],
        image_dynamic_shape[1],
        image_dynamic_shape[2],
    )

    mask_size, _ = _norm_params(mask_size, offset=None)

    half_mask_height = mask_size[0] // 2
    half_mask_width = mask_size[1] // 2

    cutout_center_height = tf.random.uniform(
        shape=[batch_size],
        minval=half_mask_height,
        maxval=image_height - half_mask_height,
        dtype=tf.int32,
        seed=seed,
    )
    cutout_center_width = tf.random.uniform(
        shape=[batch_size],
        minval=half_mask_width,
        maxval=image_width - half_mask_width,
        dtype=tf.int32,
        seed=seed,
    )

    offset = tf.transpose([cutout_center_height, cutout_center_width], [1, 0])
    return cutout(images, mask_size, offset, constant_values)


img_raw = open("./test_images/85f8cb619c66b863.jpg", 'rb').read()
image = tf.image.decode_jpeg(img_raw)
image = tf.image.convert_image_dtype(image, tf.float32)
plt.figure()
plt.imshow(image.numpy())
image = tf.image.resize(images=image, size=[600, 600])
# i1 = (image[:, :, 0] - mean[0] / 255.0) / std[0] * 255.0
# i2 = (image[:, :, 1] - mean[1] / 255.0) / std[1] * 255.0
# i3 = (image[:, :, 2] - mean[2] / 255.0) / std[2] * 255.0
# # use the all dataset data
# image = tf.concat([tf.expand_dims(i1, axis=-1), tf.expand_dims(i2, axis=-1), tf.expand_dims(i3, axis=-1)], axis=2)
# image = tf.image.per_image_standardization(image)
# 高斯噪声的标准差为 0.3
gau = tf.keras.layers.GaussianNoise(0.3)
# 以 50％ 的概率为图像添加高斯噪声
image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
# brightness随机调整
image = tf.image.random_brightness(image, 0.2)
# random left right flip
image = tf.image.random_flip_left_right(image)
# random up down flip
image = tf.image.random_flip_up_down(image)
# cutout ~2 patches / image
# width / height 20
image = tf.expand_dims(image, axis=0)
image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
image = tf.squeeze(image, axis=0)
# 随机旋转图片 0 ~ 30°
angle = tf.random.uniform([], minval=0, maxval=30)
image = tfa.image.rotate(image, angle)
image = tf.expand_dims(image, axis=0)
image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
image = tf.cond(tf.random.uniform([]) < 0.5, lambda: random_cutout(image, [20, 20]), lambda: image)
image = tf.squeeze(image, axis=0)
image = tf.image.random_jpeg_quality(image, 80, 100)
image = tf.image.random_crop(image, [512, 512, 3])
plt.figure()
print(image.shape)
plt.imshow(image.numpy())
plt.show()
