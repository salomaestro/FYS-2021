import numpy as np
import matplotlib.pyplot as plt
import os

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_seals_train = os.path.join(dirname, "seals_train.csv")
data = np.genfromtxt(filename_seals_train, delimiter=" ")
