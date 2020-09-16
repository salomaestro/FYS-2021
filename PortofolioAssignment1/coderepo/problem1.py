import numpy as np

games = np.genfromtxt("FYS-2021\PortofolioAssignment1\coderepo\chess-games.csv", delimiter=" ")
names = np.genfromtxt("FYS-2021\PortofolioAssignment1\coderepo\chess-games-names.csv", delimiter=",", dtype=str)

id, name = names.T
id = id.astype("int32")
# rows are i, columns are j. aij

A = np.zeros((np.shape(id)[0], np.shape(id)[0]))
index = np.arange(0, np.shape(games)[0])


white = games[index].T[0].astype("int32")
black = games[index].T[1].astype("int32")
score = games[index].T[2]
#når score = 1 returnerer where en index på den runden hvit vant

white_won = np.where(score == 1) # index til runden
black_won = np.where(score == 0) # index til runden
tie = np.where(score == 0.5)

print(games[white_won])
i = np.arange(0, np.shape(A)[0])
# A[rad][kolonne]


I, J, _ = games[white_won]
print(I)

print(np.count_nonzero(A))






# A[white][black] += np.where(score == 1, x=1, y=0)



for ind, _ in enumerate(games):
    white = games[ind][0].astype("int32")
    black = games[ind][1].astype("int32")
    score = games[ind][2]
    if score == 1:
        A[white][black] += 1
    elif score == 0:
        A[black][white] += 1
