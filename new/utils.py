import pandas as pd
import numpy as np
import struct
import os
import cv2 as cv


def Euclidean_dist(x, y):
    assert x.shape == y.shape
    return np.sqrt(np.sum(np.square(x-y)))


def load_binaryClassify_data(path):
    X = pd.read_csv(path, header=None)
    X = X.copy()
    y = X.pop(X.columns[-1])
    labels = y.unique()
    assert len(labels) == 2
    if -1 not in labels or 1 not in labels:
        pos_index = np.where(y == labels[0])
        neg_index = np.where(y == labels[1])
        y = np.ones(y.shape)
        y[pos_index] = 1
        y[neg_index] = -1
    return X, y


def load_mnist(path='dataset/mnist', mode='train'):
    assert mode in ['train', 'test']
    if mode == 'train':
        img_path = os.path.join(path, 'train-images-idx3-ubyte')
        label_path = os.path.join(path, 'train-labels-idx1-ubyte')
    else:
        img_path = os.path.join(path, 't10k-images-idx3-ubyte')
        label_path = os.path.join(path, 't10k-labels-idx1-ubyte')

    with open(label_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(img_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def get_HOGFeatures(trainSet):
    features = []
    winSize = (28, 28)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (4, 4)
    nbins = 9
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    for image in trainSet:
        img = np.reshape(image, (28, 28))
        cv_img = img.astype(np.uint8)
        feature = hog.compute(cv_img)
        features.append(feature)
    features = np.array(features).squeeze(axis=2)
    return features


# if __name__ == '__main__':
