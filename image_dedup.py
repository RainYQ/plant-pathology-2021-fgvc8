from imagededup.methods import PHash
from imagededup.utils import plot_duplicates
import pickle

import os


def phash_image():
    phasher = PHash()

    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(image_dir='./train_images')
    pickle.dump(encodings, open('encodings.pkl', 'wb'))
    print(encodings)

    # Find duplicates using the generated encodings
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    pickle.dump(duplicates, open('duplicates.pkl', 'wb'))
    print(duplicates)

    # plot duplicates obtained for a given file using the duplicates dictionary
    filenames = os.listdir('./train_images')
    for filename in filenames:
        plot_duplicates(image_dir='./train_images',
                        duplicate_map=duplicates,
                        filename=filename)


def main():
    train_images = os.listdir('./train_images')
    encodings = pickle.load(open('encodings.pkl', 'rb'))
    duplicates = pickle.load(open('duplicates.pkl', 'rb'))
    duplicates = set(frozenset([image] + duplicates[image]) for image in train_images if len(duplicates[image]) > 0)
    print(len(duplicates))
    with open('duplicates_others.csv', 'w') as file:
        for row in duplicates:
            file.write(','.join(row) + '\n')


if __name__ == '__main__':
    main()
