from pyqubo import Binary
import numpy as np
import neal

# from parser import parse

M = 200.0  # penalty coeff

# ## input parsing
def parse(filename):

    with open(filename) as file:
        data = file.read().split('\n')

    data = list(filter(lambda x : x.strip(), data))
    # print(data)

    data_iter = iter(data)

    n = int(next(data_iter))
    # print(n)
    f, d = [], []

    ## constructing input matrices
    for i in range(0, n):
        f.append([k for k in list(map(int, next(data_iter).split()))])

    for i in range(0, n):
        d.append([k for k in list(map(int, next(data_iter).split()))])

    return n, f, d



##### start of process #####

n, f, d = parse("qapdata/had12.dat")
print(n, f, d)

## constructing qubo matrix
qubo_mat = np.einsum("ij,kl->ikjl", f, d)
# print(qubo_mat)
qubo_mat = qubo_mat.reshape(n**2, n**2)
# print(qubo_mat)

x = [Binary("x%d"%i) for i in range(n**2)]
# print(x)

H = 0
p_terms = 0
for i in range(n**2):
    for j in range(n**2):
        H += qubo_mat[i][j]*x[i]*x[j]

# adding penalty terms
for i in range(n):
    p_terms += (x[0+i*n]+x[1+i*n]+x[2+i*n]-1)**2 + (x[0*n+i]+x[1*n+i]+x[2*n+i]-1)**2
p_terms *= M
# print(p_terms)
H += p_terms



model = H.compile()
output = model.to_qubo()
# print(output)
tups_dict = output[0].keys()

mat = [[0 for _ in range(n**2)] for _ in range(n**2)]
for x_str, y_str in tups_dict:
    x, y = int(x_str[1:]), int(y_str[1:])
    # print(x,y)
    mat[x][y] = output[0][(x_str, y_str)]


# print(mat)
# for row in mat:
#     print(row)

print("Output 1:", output[1])



## solving
bqm = model.to_bqm()
sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=10)
decoded_samples = model.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)
best = best_sample.sample
print(best)

## converting output to matrix
best_keys = best.keys()
# print(best_keys)
x_mat = [[0 for _ in range(n)] for _ in range(n)]
for x_str in best_keys:
    x = int(x_str[1:])
    val = best[x_str]
    row = x // n
    col = x % n
    # print(x, val, row, col)
    x_mat[row][col] = val

# debug prints
for i in range(n):
    print(x_mat[i])
# for i in range(n):
#     for j in range(n):
#         if (x_mat[i][j] == 1): 
#             print(i, j)



## getting objective function val
x = x_mat
obj = 0

for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                obj += f[i][j]*d[k][l]*x[i][k]*x[j][l]

print(obj)