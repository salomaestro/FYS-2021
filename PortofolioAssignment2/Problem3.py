import numpy as np
import matplotlib.pyplot as plt
import os
import sys

if sys.version_info[0] != 3:
    print("This code might not be working properly because you are running python version {}, while a python 3 distribution is preferred!".format(sys.version))

dirname = os.path.dirname(__file__)
filename_censored_data = os.path.join(dirname, "censored_data.csv")

censoredData = np.genfromtxt(filename_censored_data, delimiter=" ")
print(censoredData)
print(type(censoredData[2, 1]))
print(np.isnan(censoredData[2, 1]))

def imputation_mean(data):
    """
    Function which uses mean of data to replace nan values. Also known as imputation

    Args:
        (np.ndarray) - data with nan values.
    """
    datashape = data.shape
    meanimputated = []
    for column in data.T:
        meancol = np.ones_like(column) * np.mean(column)
        meanimputated.append(np.where(np.isnan(column), meancol, column))
    meanimputated = np.asarray(meanimputated)

    # flatData = data.flatten()
    # boolarray = np.isnan(flatData)
    # meanimputated = np.where(boolarray, np.mean(flatData), flatData)
    # unflattened = np.reshape(meanimputated, datashape)
    print(meanimputated)

imputation_mean(censoredData)
