import numpy as np
import matplotlib.pyplot as plt
import os
from ML_functions_PA import power_iteration, PageRank, ModifiedPageRank, LinearRegression

# fixing the file path
dirname = os.path.dirname(__file__)
filename_game = os.path.join(dirname, "chess-games.csv")
filename_names = os.path.join(dirname, "chess-games-names.csv")
filename_elo = os.path.join(dirname, "chess-games-elo.csv")
# creating arrays of the data we are to process
games = np.genfromtxt(filename_game, delimiter=" ")
names = np.genfromtxt(filename_names, delimiter=",", dtype=str)
elo = np.genfromtxt(filename_elo, delimiter=",", dtype=int)

#################### Problem (1f) #################################
# extracting the id, and singular names out of the names array.
id, name = names.T
id = id.astype("int32")

# constructing an matrix A of correct size and shape
A = np.zeros((np.shape(id)[0], np.shape(id)[0]))

# looping throught each game such that we see them as i.e: [0, 1, 0.0] as the first game,
# with index: ind.
for ind, _ in enumerate(games):
    # assigning rows of the playedGames to white, black and the score
    white = games[ind][0].astype("int32")
    black = games[ind][1].astype("int32")
    score = games[ind][2]
    # if white loses assign one loss to whites row, index black
    if score == 0:
        A[white][black] += 1
    # do the opposite for blacks row, index white
    elif score == 1:
        A[black][white] += 1

# dividing the elements of each row by the sum of the row
A = A / np.sum(A, axis=1).reshape((-1, 1))

# using the page rank method of the imported python-file: ML_functions_PA.py
rank = PageRank(A.T)

def RankChessGames(ranks):
    # flip to get the desired output at the top of the array
    ranks = np.flip(ranks)

    # pick out only the top ten id's
    rankid = ranks[0:10]

    # find the top ten id's in usedNames
    idname = names[rankid]

    # Cute print ;)
    string = ""
    string += "rank, id, name"
    for i in range(len(idname)):
        string += "\n" + str(i + 1) + ", " + str(idname[i][0]) + ", " + str(idname[i][1])
    return string

result = RankChessGames(rank)
# print(result)

################################# Problem (1g) ###############################################

def test_stochasticy():
    """
    Let's test the stochastic
    """
    for row, _ in enumerate(A):
        if A[row].all():
            print("oh shit!")
# test_stochasticy()

# To make sure our matrix is irreducible, we take the google matrix appoach, using the damping factor alpha = 0.85
AModified = RankChessGames(PageRank(ModifiedPageRank(A.T)))

# print("\n\nResults from ranking with the Google Matrix approach \n", AModified)
# print("We see that now we eliminate the instances where some had the same scores, such that all are ranked in relation to each other.")

################################# Problem (1h) ###############################################
# I now use the modified matrix's eigenvector that's yielded fra the power iteration method.
rank_score = power_iteration(ModifiedPageRank(A.T), 100)

# Since the last element is just an extra page we use to modify this matrix to an Google matrix and we know that our ModifiedPageRank method only concatenates this last page to the end of the matrix, then the last element of the eigenvector rank_score will belong to this extra page. We need to remove it since we only have x-numbers of elo scores, and x+1 numbers of rows in the eigenvector (columnvector).
rank_score = rank_score.T[:-1].T

# Since we in the power_iteration method have to convert the array to an numpy matrix object to be able to do take the power of the matrix we now need to convert the elo ndarray to an numpy matrix object.
elo = np.matrix(elo)

# Taking the log of the array elementwise.
rank_scoreLog = np.log(rank_score)

input = np.concatenate((rank_scoreLog, elo.T[1]))
linregress = LinearRegression(input.T)
b0, b1 = linregress.estimators()
print("We get the estimates: betta_0 = {:.3f}, betta_1 = {:.3f}. Where betta_0 is the y-intercept, that is if we were to draw a regression line, it would be where the line crosses the y-axis. Then since we use the form of the regression line to be y = b_0 + b_1 * x, this means b_1 is the slope of the line.".format(b0, b1))

xval, yval = linregress.least_squares()

# Have to cast theese to numpy.ndarrays for the plots
xval = np.array(xval)
yval = np.array(yval)
rank_scoreLog = np.array(rank_scoreLog)
elo_score = np.array(elo.T[1])

meansquarederror = linregress.meanSquaredError()
rSquared = linregress.r_squared()
print("The mean squared error: {:.3f}, R squared: {:.3f}.".format(meansquarederror, rSquared))

# plots
regressionline, = plt.plot(xval[0], yval[0], "g-", linewidth=2)
datapoints = plt.scatter(rank_scoreLog, elo_score, marker=".", s=20)

# for the plots
plt.xlabel("log(PageRank)")
plt.ylabel("Elo score")
plt.title("Plot of Elo score against the logarithmic PageRanked chess players.\nWe get mean squared error: {:.3f}, R^2: {:.3f}".format(meansquarederror, rSquared))
plt.legend((regressionline, datapoints), ("Regression line", "Datapoints"), loc="upper left", fancybox=True, shadow=True)
plt.show()

# picture = os.path.join(dirname, "Plot_Logarithmic.pdf")
# plt.savefig(picture)
