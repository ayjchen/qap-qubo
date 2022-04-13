f = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
d = [[0, 8, 15], [8, 0, 13], [15, 13, 0]]
x = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
obj = 0

for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                obj += f[i][j]*d[k][l]*x[i][k]*x[j][l]

print(obj)