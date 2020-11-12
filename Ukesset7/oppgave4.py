import numpy as np
import os
import matplotlib.pyplot as plt
from Problem1 import LogisticDiscrimination
from dataloader import load_data

def importdata():
    optdigits = load_data("optdigits-012.csv")
    return optdigits

class MultivarLogisticDiscrimination:
    def __init__(self, trainingdata):
        self.traindata = np.delete(trainingdata, 0, axis=1)
        self.traindata = self.traindata / np.linalg.norm(self.traindata)
        self.gt = trainingdata[:, 0]

        self.K = len(np.unique(self.gt)) - 1

    def fit(self, s=0.01, threshold=100, epochs=1000):
        self.w = np.random.uniform(-0.01, 0.01, size=(self.K, self.traindata.shape[1]))

        old_Dw = np.zeros_like(self.w)
        new_Dw = np.ones_like(self.w) * 1E2
        # while np.sum(np.abs(old_Dw - new_Dw)) > threshold:
        for _ in range(epochs):
            # print(np.sum(np.abs(old_Dw - new_Dw)))
            old_Dw = new_Dw
            y = np.exp(np.matmul(self.w, self.traindata.T)) / np.sum(np.exp(np.matmul(self.w, self.traindata.T)))

            new_Dw = old_Dw + np.matmul((self.gt - y), self.traindata)
            self.w += s * new_Dw

        print(self.w)


    def test(self):
        pass


    def accuracy(self):
        pass



def main():
    optdigits = importdata()

    test = MultivarLogisticDiscrimination(optdigits)
    test.fit(epochs=100)


    # opt = LogisticDiscrimination(optdigits)
    # _ = opt.train(2, 0.1)
    # print(_)


if __name__ == "__main__":
    main()
