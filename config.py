import numpy as np

label2id = {
    'scab': 0,
    'healthy': 1,
    'frog_eye_leaf_spot': 2,
    'rust': 3,
    'complex': 4,
    'powdery_mildew': 5,
    'scab frog_eye_leaf_spot': 6,
    'scab frog_eye_leaf_spot complex': 7,
    'frog_eye_leaf_spot complex': 8,
    'rust frog_eye_leaf_spot': 9,
    'rust complex': 10,
    'powdery_mildew complex': 11
}

label2array = {
    0: np.array([1, 0, 0, 0, 0], dtype=np.float32),
    1: np.array([0, 0, 0, 0, 0], dtype=np.float32),
    2: np.array([0, 1, 0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 1, 0, 0], dtype=np.float32),
    4: np.array([0, 0, 0, 1, 0], dtype=np.float32),
    5: np.array([0, 0, 0, 0, 1], dtype=np.float32),
    6: np.array([1, 1, 0, 0, 0], dtype=np.float32),
    7: np.array([1, 1, 0, 1, 0], dtype=np.float32),
    8: np.array([0, 1, 0, 1, 0], dtype=np.float32),
    9: np.array([0, 1, 1, 0, 0], dtype=np.float32),
    10: np.array([0, 0, 1, 1, 0], dtype=np.float32),
    11: np.array([0, 0, 0, 1, 1], dtype=np.float32)
}

id2index_list = {
    0: [0],
    1: [],
    2: [1],
    3: [2],
    4: [3],
    5: [4],
    6: [0, 1],
    7: [0, 1, 3],
    8: [1, 3],
    9: [1, 2],
    10: [2, 3],
    11: [3, 4]
}

id2array = {
    0: np.array([1, 0, 0, 0, 0], dtype=np.float32),
    1: np.array([0, 0, 0, 0, 0], dtype=np.float32),
    2: np.array([0, 1, 0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 1, 0, 0], dtype=np.float32),
    4: np.array([0, 0, 0, 1, 0], dtype=np.float32),
    5: np.array([0, 0, 0, 0, 1], dtype=np.float32),
    6: np.array([1, 1, 0, 0, 0], dtype=np.float32),
    7: np.array([1, 1, 0, 1, 0], dtype=np.float32),
    8: np.array([0, 1, 0, 1, 0], dtype=np.float32),
    9: np.array([0, 1, 1, 0, 0], dtype=np.float32),
    10: np.array([0, 0, 1, 1, 0], dtype=np.float32),
    11: np.array([0, 0, 0, 1, 1], dtype=np.float32)
}

id2array_with_healthy = {
    0: np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
    1: np.array([0, 0, 0, 0, 0, 1], dtype=np.float32),
    2: np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
    3: np.array([0, 0, 1, 0, 0, 0], dtype=np.float32),
    4: np.array([0, 0, 0, 1, 0, 0], dtype=np.float32),
    5: np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),
    6: np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
    7: np.array([1, 1, 0, 1, 0, 0], dtype=np.float32),
    8: np.array([0, 1, 0, 1, 0, 0], dtype=np.float32),
    9: np.array([0, 1, 1, 0, 0, 0], dtype=np.float32),
    10: np.array([0, 0, 1, 1, 0, 0], dtype=np.float32),
    11: np.array([0, 0, 0, 1, 1, 0], dtype=np.float32)
}

cfg = {
    'data_params': {
        'img_shape': (512, 512),
        'over_bound_img_shape': (600, 600),
        'test_img_shape': (512, 512),
        'class_type': 5
    },
    'model_params': {
        'batchsize_per_gpu': 128,
        'iteration_per_epoch': 128,
        'batchsize_in_test': 128,
        'epoch': 50,
        'mix-up': True,
        'standardization': False,
        'random_resize': True
    }
}

classes = np.array([
    'scab',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew'])

classes_extra = np.array([
    'healthy',
    'multiple_diseases',
    'rust',
    'scab'])

classes_with_healthy = np.array([
    'scab',
    'frog_eye_leaf_spot',
    'rust',
    'complex',
    'powdery_mildew',
    'healthy'])

# id2label用于输入0-11, 查找label原始名称
id2label = dict([(value, key) for key, value in label2id.items()])
# print(id2label)
