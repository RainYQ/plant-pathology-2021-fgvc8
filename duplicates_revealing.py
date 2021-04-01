from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imagehash
import PIL
import os
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

CFG = {
    'threshold': .9,
    'img_size': (512, 512)
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


SEED = 1998

seed_everything(SEED)
TRAIN_IMAGE_ROOT = './train_images'

# paths = os.listdir(TRAIN_IMAGE_ROOT)

df = pd.read_csv('./train.csv', index_col='image')

# for path in tqdm(paths):
#     image = tf.io.read_file(os.path.join(root, path))
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [CFG['img_size'][0], CFG['img_size'][1]])
#     image = tf.cast(image, tf.uint8).numpy()
#     plt.imsave(os.path.join("./hash_images", path), image)

hash_functions = [
    imagehash.average_hash,
    imagehash.phash,
    imagehash.dhash,
    imagehash.whash
]

image_ids = []
hashes = []

paths = tf.io.gfile.glob('./hash_images/*.jpg')

for path in tqdm(paths, total=len(paths)):
    image = PIL.Image.open(path)
    hashes.append(np.array([x(image).hash for x in hash_functions]).reshape(-1, ))
    image_ids.append(os.path.basename(path))

hashes = np.array(hashes)
image_ids = np.array(image_ids)

duplicate_ids = []

for i in tqdm(range(len(hashes))):
    similarity = (hashes[i] == hashes).mean(axis=1)
    duplicate_ids.append(list(image_ids[similarity > CFG['threshold']]))

duplicates = [frozenset([x] + y) for x, y in zip(image_ids, duplicate_ids)]
duplicates = set([x for x in duplicates if len(x) > 1])

print(f'Found {len(duplicates)} duplicate pairs:')
for row in duplicates:
    print(', '.join(row))
print('Writing duplicates to "duplicates.csv".')
with open('duplicates.csv', 'w') as file:
    for row in duplicates:
        file.write(','.join(row) + '\n')
for row in duplicates:

    figure, axes = plt.subplots(1, len(row), figsize=[5 * len(row), 5])
    pair = ''
    for i, image_id in enumerate(row):
        image = plt.imread(os.path.join('./hash_images', image_id))
        axes[i].imshow(image)
        axes[i].set_title(df.loc[image_id, 'labels'])
        axes[i].axis('off')
        pair += image_id
        pair += '_'
    plt.savefig('./duplicates_images/' + pair[:-1] + '.png')

    # plt.show()
