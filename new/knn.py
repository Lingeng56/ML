import numpy as np
from utils import load_mnist, get_HOGFeatures, Euclidean_dist
from collections import defaultdict
from tqdm import tqdm


class KdNode:

    def __init__(self, elem, split_dim, left, right, label):
        self.elem = elem
        self.split_dim = split_dim
        self.left = left
        self.right = right
        self.label = label


def find_nearest(node, feature, K, nodes=None, dist_fn=Euclidean_dist):
    if nodes is None:
        nodes = []
    if node is None:
        return nodes
    if len(nodes) < K:
        nodes.append((node, dist_fn(node.elem, feature)))
        nodes = sorted(nodes, key=lambda x: x[1])
    else:
        if dist_fn(node.elem, feature) < nodes[-1][1]:
            nodes.pop(-1)
            nodes.append((node, dist_fn(node.elem, feature)))
            nodes = sorted(nodes, key=lambda x: x[1])
    split = node.split_dim
    if feature[split] <= node.elem[split]:
        nodes = find_nearest(node.left, feature, K, nodes)
        if len(nodes) < K or abs(feature[split] - node.elem[split]) < dist_fn(node.elem, feature):
            nodes = find_nearest(node.right, feature, K, nodes)
    else:
        nodes = find_nearest(node.right, feature, K, nodes)
        if len(nodes) < K or abs(feature[split] - node.elem[split]) < dist_fn(node.elem, feature):
            nodes = find_nearest(node.left, feature, K, nodes)

    return nodes


def count_max(labels):
    label2num = defaultdict(int)
    for label in labels:
        label2num[label] += 1
    return sorted(label2num.items(), key=lambda x: x[1])[-1][0]


class KNN:

    def __init__(self, K, dist=Euclidean_dist):
        self.root = None
        self.K = K
        self.dist = dist

    def fit(self, X, y):
        features = get_HOGFeatures(X)
        self.root = self.create_KDTree(features, y)

    def create_KDTree(self, features, y, split_dim=1, depth=1):
        if features.shape[0] == 0:
            return None
        k = features.shape[1]
        sort_index = features[:, split_dim].argsort()
        features = features[sort_index]
        labels = y[sort_index]
        split_pos = len(features) // 2
        split_next = depth % k + 1
        return KdNode(features[split_pos],
                      split_dim,
                      self.create_KDTree(features[:split_pos, :], labels[:split_pos], split_next, depth + 1),
                      self.create_KDTree(features[split_pos + 1:, :], labels[split_pos + 1:], split_next, depth + 1),
                      labels[split_pos])

    def predict(self, testData):
        res = []
        testData = get_HOGFeatures(testData)
        print(testData.shape)
        for image in tqdm(testData):
            nodes = find_nearest(self.root, image, self.K)
            labels = [node.label for node, _ in nodes]
            res.append(count_max(labels))

        return np.array(res)


def accuracy(pre, tru):
    return sum(pre == tru) / pre.shape[0]


if __name__ == '__main__':
    model = KNN(10)
    train = load_mnist()
    test = load_mnist(mode='test')
    model.fit(*train)
    preds = model.predict(test[0])
    print(accuracy(preds, test[1]))
