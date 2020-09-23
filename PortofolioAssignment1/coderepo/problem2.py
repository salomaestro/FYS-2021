import numpy as np
import os
import matplotlib.pyplot as plt
from ML_functions_PA import ProbabilityDistributions, Bayes_classifier

dirname = os.path.dirname(__file__)
filename_train = os.path.join(dirname, "optdigits-1d-train.csv")

trainingset = np.genfromtxt(filename_train, delimiter=" ")

# Here we filter out the appropriate information and put the x values which correspond to an label = (0/1)
train0 = trainingset.T[1][np.where(trainingset.T[0] == 0)[0]]
train1 = trainingset.T[1][np.where(trainingset.T[0] == 1)[0]]
train = trainingset.T[1]
n = trainingset.T[0].shape[0]

# Are given alpha = 9
alpha = 9

# Pretty much just using the given formuale
n0 = train0.shape[0]
betta_hat = 1 / (n0 * alpha) * np.sum(train0)

n1 = train1.shape[0]
my_hat = 1 / n1 * np.sum(train1)
sigma2_hat = 1 / n1 * np.sum((train1 - my_hat) ** 2)

# choose C = 1 if P(C = 1 | x0, x1) > P(C = 0 | x0, x1) choose C = 0 otherwise. P(C|x0 ,x1) = P(C)p(x0, x1|C)/p(x0, x1)

# prior probabilities
priorC0 = n0 / n
priorC1 = n1 / n

train0Dist = ProbabilityDistributions(train0)
train1Dist = ProbabilityDistributions(train1)
gammadist = np.array(train0Dist.gamma(alpha, betta_hat)).T
normaldist = np.array(train1Dist.normal()).T

#sorting by the first column to be able to plot with lines
gammadist_sorted = gammadist[gammadist[:,0].argsort()].T
normaldist_sorted = normaldist[normaldist[:,0].argsort()].T

def plot():
    # SIMPLE CODE FOR PLOTTING
    # set density to true, such that the area under the curve is equal to 1.
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    axs[0].hist(train0, bins=50, density=True, color="pink", label="Gamma histogram")
    axs[0].plot(gammadist_sorted[0], gammadist_sorted[1], "red", label="Gamma distribution")
    axs[0].set_title("Gamma")

    axs[1].hist(train1, bins=50, density=True, color="turquoise", label="Normal histogram")
    axs[1].plot(normaldist_sorted[0], normaldist_sorted[1], "blue", label="Normal distribution")
    axs[1].set_title("Normal")

    axs[2].hist(train0, bins=50, density=True, color="pink", label="Gamma histogram")
    axs[2].plot(gammadist_sorted[0], gammadist_sorted[1], "red", label="Gamma distribution")
    axs[2].hist(train1, bins=50, density=True, color="turquoise", label="Normal histogram")
    axs[2].plot(normaldist_sorted[0], normaldist_sorted[1], "blue", label="Normal distribution")
    axs[2].set_title("Both")
    axs[2].legend(bbox_to_anchor=(.005, 1.3, .5, .5))

    plt.subplots_adjust(hspace=0.5)

    fig.suptitle("Histogram and estimated deistributions for:")
    plt.show()

#################### (2c) ##############################
# Classifying
class0, class1 = Bayes_classifier(train)







def main():
    print("We get the point estimates: betta_hat = {:.5f}, my_hat = {:.5f}, sigma2_hat = {:.5f}".format(betta_hat, my_hat, sigma2_hat))
    # plot()

if __name__ == "__main__":
    main()
