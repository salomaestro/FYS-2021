import os
import numpy as np

def load_data(*filenames, delimiter=" "):
    dirname = os.path.dirname(__file__)
    data = []
    for filename in filenames:
        filename_joined = os.path.join(dirname, str(filename))
        data.append(np.genfromtxt(filename_joined, delimiter=delimiter))
    if len(data) == 1:
        return np.asarray(data[0])
    else:
        return np.asarray(data)
