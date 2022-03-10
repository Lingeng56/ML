import numpy as np
from tqdm import tqdm
from utils import load_binaryClassify_data


class Perceptron:

    def __init__(self, epochs=20, lr=0.1, random_state=0):
        self.weight = None
        self.lr = lr
        self.epochs = epochs
        np.random.seed(random_state)

    def fit(self, X, y):
        self.weight = np.random.random((X.shape[1] + 1, 1))
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        for _ in tqdm(range(self.epochs)):
            for i in range(X.shape[0]):
                if np.matmul(self.weight.T, X[i:i + 1, :].T).item() * y[i] < 0:
                    self.weight += self.lr * X[i:i + 1, :].T * y[i]

        print("Final Weight: ", self.weight)

    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)


if __name__ == '__main__':
    model = Perceptron()
    X, y = load_binaryClassify_data('dataset/binary_classify/sonar.csv')
    model.fit(X, y)
