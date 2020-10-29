import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil
import pandas as pd
from ML_functions_PA import LinearRegression

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_censored_data = os.path.join(dirname, "censored_data.csv")
filename_uncensored_data = os.path.join(dirname, "uncensored_data.csv")

censoredData = np.genfromtxt(filename_censored_data, delimiter=" ")
uncensoredData = np.genfromtxt(filename_uncensored_data, delimiter=" ")
censoredData_df = pd.DataFrame(censoredData)

def imputationByRegression(df, actualdata, columnIndexOfNan, plot=False):
    """
    Function which fills nan values by regression.

    Args:
        param1: (np.ndarray) - data, containing values and nan's.
        param2: (np.ndarray) - ground truth, to compare against
    """
    corrmat = df.corr().to_numpy()
    minvalue = np.min(corrmat[:, columnIndexOfNan])
    minvalueind = np.where(corrmat[:, columnIndexOfNan] == minvalue)[0][0]
    data = df.to_numpy()

    regressiondata = np.delete(data, minvalueind, axis=1)
    index_of_nan = np.where(np.isnan(regressiondata[:, columnIndexOfNan]))
    regressiondata_except_nan = np.delete(regressiondata, index_of_nan, axis=0)

    model = LinearRegression(regressiondata_except_nan)
    x, y = model.least_squares()

    nanrows = regressiondata[index_of_nan]
    nanUncensored = actualdata[index_of_nan][:, 1]

    estimated_nan = model.w1 * nanrows[:, 0] + model.w0

    MSE_regression_estimator = np.mean((nanUncensored - estimated_nan) ** 2)

    if plot == True:
        plt.plot(x, y)
        plt.scatter(regressiondata_except_nan[:, 0], regressiondata_except_nan[:, 1])
        plt.show()

    return MSE_regression_estimator, estimated_nan


# Finds the mean along the column axis
mean_column = np.nanmean(censoredData, axis=0)
median_column = np.nanmedian(censoredData, axis=0)
max_column = np.nanmax(censoredData, axis=0)

# Replaces nan's with the mean, median and max values. The true argument specifies that nan_to_num does not change censoredData array, but rater store the result in the ximputated variable.
meanimputated = np.nan_to_num(censoredData, True, mean_column)
medianimputated = np.nan_to_num(censoredData, True, median_column)
maximputated = np.nan_to_num(censoredData, True, max_column)

# Extract middle column
meanimputated = meanimputated[:, 1]
medianimputated = medianimputated[:, 1]
maximputated = maximputated[:, 1]

# Make the matrix of data collapse into a row containing all values along same axis.
flattened_censoredData = censoredData.flatten()

# Finds the index of what datapoint the nan value is located, in the flattened array, since this dataset only has nan values along the middle column, dividing the result by 3 then rounding and making sure index values are correct datatype will give correct index.
index_of_nan = np.round(np.asarray(np.where(np.isnan(flattened_censoredData))) / 3)[0].astype("int64")

uncensored_nan_values = uncensoredData[index_of_nan]
uncensored_nan_values = uncensored_nan_values[:, 1]

# Mean Sqared Error between the ground truth and the estimator at the datapoints.
MSE_mean_estimator = np.mean((uncensored_nan_values - meanimputated[index_of_nan]) ** 2)

MSE_median_estimator = np.mean((uncensored_nan_values - medianimputated[index_of_nan]) ** 2)

MSE_max_estimator = np.mean((uncensored_nan_values - maximputated[index_of_nan]) ** 2)

MSE_regression_estimator, pred_val = imputationByRegression(censoredData_df, uncensoredData, 1)

def main():
    print("The columns which hosted nan values is the middle column. Here we have the mean, median and max values:\nmean = {0}, \nmedian = {1}, \nmax = {2}\nMSE for mean estimator: {3}\nMSE for median estimator: {4}\nMSE for max estimator: {5} \nMSE for regression estimator: {6}.\n\nPredicted values for nan's are:\n {7}".format(mean_column[1], median_column[1], max_column[1], MSE_mean_estimator, MSE_median_estimator, MSE_max_estimator, MSE_regression_estimator, pred_val))


if __name__ == "__main__":
    main()
