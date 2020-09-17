import numpy as np
import os
from ML_functions_PA import power_iteration, PageRank

#creating arrays of the data we are to process
games = np.genfromtxt("coderepo/chess-games.csv", delimiter=" ")
names = np.genfromtxt("coderepo/chess-games-names.csv", delimiter=",", dtype=str)

# extracting the id, and singular names out of the names array.
id, name = names.T
id = id.astype("int32")

#constructing an matrix A of correct size and shape
A = np.zeros((np.shape(id)[0], np.shape(id)[0]))

#looping throught each game such that we see them as i.e: [0, 1, 0.0] as the first game,
# with index: ind.
for ind, _ in enumerate(games):
    #assigning
     white = games[ind][0].astype("int32")
     black = games[ind][1].astype("int32")
     score = games[ind][2]
     #if white loses assign one loss to whites row, index black
     if score == 0:
         A[white][black] += 1
    #do the opposite for blacks row, index white
     elif score == 1:
         A[black][white] += 1

#dividing the elements of each row by the sum of the row
A = A / np.sum(A, axis=1).reshape((-1, 1))

ranks = PageRank(A.T)
ranks = np.flip(ranks)
rankid = ranks[0:10]

idname = names[rankid]

crr_shape = np.shape(idname[1])[0]
list_of = np.arange(1, crr_shape + 1).astype("str")

def prettyprint():
    string = ""
    string += "rank, id, name"
    for i in range(len(idname)):
        string += "\n" + str(i + 1) + ", " + str(idname[i][0]) + ", " + str(idname[i][1])
    print(string)
prettyprint()









#A = np.zeros((np.shape(id)[0], np.shape(id)[0]))
#index = np.arange(0, np.shape(games)[0])
#white_won = np.where(games.T[2] == 1) # index of the current game
#black_won = np.where(games.T[2] == 0) # index of the current game
#tie = np.where(games.T[2] == 0.5)
#i, j, _ = games[white_won].T.astype("int32")
#k, l, _ = games[black_won].T.astype("int32")
#print(i, k)
#print(np.size(i), "size", np.size(k))
#A[j][i] += 1
#A[k][l] += 1
#print(np.sum(A))
#print(A[171, 115])
#print(A)
#print(A[115, 171])
#print(A[171, 115])
#A / np.sum(A, axis=1)
#print(np.sum(A, axis=1))
######################
#Still needs a way to divide each row on the total losses of the row.
######################
# print(A[1] / np.sum(A[1]))
#mat_ind = np.arange(0, np.shape(A)[0])
#print(mat_ind.shape)
#A[mat_ind] = A[mat_ind] / np.sum(A[mat_ind])
# print(A)
# print(np.shape(mat_ind))
# A[games[white_won]] += 1
#print(np.count_nonzero(A))
