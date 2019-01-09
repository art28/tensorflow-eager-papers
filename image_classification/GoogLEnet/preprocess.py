import pickle
import numpy as np
from glob import glob
import os


def load_batches(file_name):
    with open(file_name, 'rb') as f:
        dictionary = pickle.load(f, encoding='latin1')
    return dictionary


def get_X(dictionary):
    return dictionary['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def get_y(dictionary):
    return dictionary['labels']


def preprocess(filelist):
    X = list()
    y = list()
    for batch_file in filelist:
        dictionary = load_batches(batch_file)
        X.append(get_X(dictionary))
        y += (get_y(dictionary))

    X = np.concatenate(X, axis=0).astype(np.float32)
    X = X / 255.
    y = np.array(y)

    return X, y


def prerprocess_train(dir_name):
    return preprocess(glob(os.path.join(dir_name, '*_batch_*')))


def prerprocess_test(dir_name):
    return preprocess(glob(os.path.join(dir_name, 'test_batch')))
