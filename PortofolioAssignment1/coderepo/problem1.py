import numpy as np

games = np.genfromtxt("FYS-2021\PortofolioAssignment1\coderepo\chess-games.csv", delimiter=" ")
names = np.genfromtxt("FYS-2021\PortofolioAssignment1\coderepo\chess-games-names.csv", delimiter=",", dtype=str)

id, name = names.T
id = id.astype("int32")
# rows are i, columns are j. aij


A = np.zeros((np.shape(id)[0], np.shape(id)[0]))
index = np.arange(0, np.shape(games)[0])


white_won = np.where(games.T[2] == 1) # index til runden
black_won = np.where(games.T[2] == 0) # index til runden
tie = np.where(games.T[2] == 0.5)

total_of_rows_w = np.zeros(np.shape(id))


i, j, _ = games[white_won].T.astype("int32")
print(i)


k, l, _ = games[black_won].T.astype("int32")


A[i, j] += 1
A[l, k] += 1

# print(A[1] / np.sum(A[1]))
mat_ind = np.arange(0, np.shape(A)[0])
A[mat_ind] = A[mat_ind] / np.sum(A[mat_ind])
# print(A)



# print(np.shape(mat_ind))
# A[games[white_won]] += 1
print(np.count_nonzero(A))





# for ind, _ in enumerate(A):
#     ind = ind[0]
#     A[ind]




#
# A = np.zeros((np.shape(id)[0], np.shape(id)[0]))
#
# for ind, _ in enumerate(games):
#     white = games[ind][0].astype("int32")
#     black = games[ind][1].astype("int32")
#     score = games[ind][2]
#     if score == 1:
#         A[white][black] += 1
#     elif score == 0:
#         A[black][white] += 1
# print(A)
#
# print(np.count_nonzero(A))
