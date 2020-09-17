import numpy as np
import os
from ML_functions_PA import power_iteration, PageRank, ModifiedPageRank

# fixing the file path
dirname = os.path.dirname(__file__)
filename_game = os.path.join(dirname, "chess-games.csv")
filename_names = os.path.join(dirname, "chess-games-names.csv")

# creating arrays of the data we are to process
games = np.genfromtxt(filename_game, delimiter=" ")
names = np.genfromtxt(filename_names, delimiter=",", dtype=str)

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
print(result)

################################# Problem (1g) ###############################################

def test_stochasticy():
    """
    Let's test the stochastic
    """
    for row, _ in enumerate(A):
        if A[row].all():
            print("oh shit!")
test_stochasticy()

# To make sure our matrix is irreducible, we take the google matrix appoach, using the damping factor alpha = 0.85

print("\n\nResults from ranking with the Google Matrix approach \n", RankChessGames(ModifiedPageRank(A.T)))
print("We see that now we eliminate the instances where some had the same scores, such that all are ranked in relation to each other.")

################################# Problem (1h) ###############################################
#
