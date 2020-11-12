import numpy as np
import os
import matplotlib.pyplot as plt
from Problem1 import LogisticDiscrimination
from dataloader import load_data

def importdata():
    optdigits = load_data("optdigits-012.csv")
    return optdigits

def subset_selection(data):
    classes = np.unique(data[:,0 ])

    for num in classes:
        usedata = data[np.where(data[:, 0] == num)]
        temp = LogisticDiscrimination(usedata)
        res = temp.train(8, 0.1)
        print(res)


def main():
    optdigits = importdata()

    subset_selection(optdigits)

    # opt = LogisticDiscrimination(optdigits)
    # _ = opt.train(2, 0.1)
    # print(_)


if __name__ == "__main__":
    main()
