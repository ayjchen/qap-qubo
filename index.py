from cmath import exp
from pyqubo import Binary
import numpy as np
import neal

import input_parser as prs

M = 20000.0  # penalty coeff
filename = "nug12"


##### start of process #####
n, f, d = prs.parse_in("qapdata/" + filename + ".dat")
# print(n, f, d)

## constructing qubo matrix ############
qubo_mat = np.einsum("ij,kl->ikjl", f, d)
qubo_mat = qubo_mat.reshape(n**2, n**2)

x = [Binary("x%d"%i) for i in range(n**2)]
H = 0
p_terms = 0
for i in range(n**2):
    for j in range(n**2):
        H += qubo_mat[i][j]*x[i]*x[j]

## adding penalty terms ###########
for i in range(n):
    # p_terms += (x[0+i*n]+x[1+i*n]+x[2+i*n]-1)**2 + (x[0*n+i]+x[1*n+i]+x[2*n+i]-1)**2
    p_terms += (sum([x[j+i*n] for j in range(n)]) - 1)**2 + (sum([x[j*n+i] for j in range(n)]) - 1)**2
p_terms *= M
# print("pterms:", p_terms)
H += p_terms


## compiling qubo model ###########
model = H.compile()
output = model.to_qubo()
tups_dict = output[0].keys()
print("Penalty additive constant:", output[1])

### qubo matrix ###
# mat = [[0 for _ in range(n**2)] for _ in range(n**2)]
# for x_str, y_str in tups_dict:
#     x, y = int(x_str[1:]), int(y_str[1:])
#     # print(x,y)
#     mat[x][y] = output[0][(x_str, y_str)]

# print(mat)
# for row in mat:
#     print(row)
### END qubo matrix ###





## solving ############
bqm = model.to_bqm()
sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=10)
decoded_samples = model.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)
best = best_sample.sample

## converting output to matrix
best_keys = best.keys()
x_mat = [[0 for _ in range(n)] for _ in range(n)]
for x_str in best_keys:
    x = int(x_str[1:])
    val = int(best[x_str])
    row = x // n
    col = x % n
    x_mat[row][col] = val

# debug prints
pos = []
for i in range(n):
    print(x_mat[i])
    pos.append(x_mat[i].index(1)+1)




## getting objective function val
x = x_mat
obj = 0

for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                obj += f[i][j]*d[k][l]*x[i][k]*x[j][l]

expected_obj, expected_pos = prs.parse_out("qapsoln/" + filename + ".sln")

print("Expected positions:", expected_pos)
print("Actual positions:", pos)
print("Expected opt obj:", expected_obj, "\nActual output obj:", obj)