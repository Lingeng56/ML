import numpy as np
import matplotlib.pyplot as plt


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data(file_name):
    data = np.loadtxt(file_name)
    dataset = []
    for i in range(len(data)):
        dataset.append([1, data[i, 0], data[i, 1], data[i, 2]])
    return np.array(dataset)


class Logistic:
    def __init__(self):
        self.w = np.ones(shape=(3,))

    def train(self, dataset, epochs=50, lr=0.001):
        y = dataset[:, -1]
        print("shape(y)", np.shape(y))
        data = dataset[:, :-1]
        print("shape(data)", np.shape(data))
        print("shape(self.w", np.shape(self.w))
        for epoch in range(epochs):
            pre = Sigmoid(np.matmul(data, self.w))
            loss = pre - y
            self.w = self.w - lr * np.matmul(data.T, loss)
        self.show_img(dataset, epochs)

    def test(self, dataset):
        y = dataset[:, -1]
        data = dataset[:, :-1]
        h = Sigmoid(np.matmul(data, self.w))
        h[h > 0.5] = 1
        h[h < 0.5] = 0
        error = 0
        result = h
        for i in range(len(result)):
            if result[i] != y[i]:
                error += 1
        self.show_img(dataset,0)
        print(" the error is {}%".format(100 * error / float(len(result))))

    def show_img(self, dataset, i):
        x = np.arange(-3.0, 3.0, 0.1)
        x2 = (-self.w[0] - self.w[1] * x) / self.w[2]
        plt.scatter([dataset[dataset[:, -1] == 0][:, 1]], [dataset[dataset[:, -1] == 0][:,2]], edgecolors='red',
                    marker='o', linewidths=3)
        plt.scatter([dataset[dataset[:, -1] == 1][:, 1]], [dataset[dataset[:, -1] == 1][:,2]], edgecolors='green',
                    marker='o', linewidths=3)
        plt.plot(x, x2)
        plt.title("迭代第{}次".format(i))
        plt.show()


if __name__ == '__main__':
    file_name = "./datasets/logisticdata.txt"
    dataset = load_data(file_name)
    print(np.shape(dataset))
    train_data = dataset[:80]
    test_data = dataset[80:]
    print(len(train_data))
    model = Logistic()
    model.train(train_data, epochs=500)
    model.test(test_data)
