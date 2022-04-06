from pyqubo import Binary
import numpy as np
import neal

M = 200.0  # penalty coeff

## input parsing
n = input()
n = int(n)
f = []
d = []

## constructing input matrices
for i in range(0, n):
    f.append([k for k in list(map(int, input().split()))])

for i in range(0, n):
    d.append([k for k in list(map(int, input().split()))])

print(f, d)

## constructing qubo matrix
qubo_mat = np.einsum("ij,kl->ikjl", f, d)
qubo_mat = qubo_mat.reshape(n**2, n**2)
print(qubo_mat)

x = [Binary("x%d"%i) for i in range(n**2)]
print(x)

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
# p_terms = M*((x[0]+x[1]+x[2]-1)**2 + (x[3]+x[4]+x[5]-1)**2 + (x[6]+x[7]+x[8]-1)**2 + (x[0]+x[3]+x[6]-1)**2 + (x[1]+x[4]+x[7]-1)**2 + (x[2]+x[5]+x[8]-1)**2)
H += p_terms

######## bruteforce

# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18 = Binary('x1'), Binary('x2'), Binary('x3'), Binary('x4'), Binary('x5'), Binary('x6'), Binary('x7'), Binary('x8'), Binary('x9'), Binary('x10'), Binary('x11'), Binary('x12'), Binary('x13'), Binary('x14'), Binary('x15'), Binary('x16'), Binary('x17'), Binary('x18')
# p_terms = M*((x1+x2+x3-1)**2 + (x4+x5+x6-1)**2 + (x7+x8+x9-1)**2 + (x1+x4+x7-1)**2 + (x2+x5+x8-1)**2 + (x3+x6+x9-1)**2)
# H = 80*x1*x5 + 150*x1*x6 + 32*x1*x8 + 60*x1*x9 + 80*x2*x4 + 130*x2*x6 + 60*x2*x7 + 52*x2*x9 + \
#     150*x3*x4 + 130*x3*x5 + 60*x3*x7 + 52*x3*x8 + 48*x4*x8 + 90*x4*x9 + 78*x5*x9 + 78*x6*x8 + p_terms
########

model = H.compile()
output = model.to_qubo()
# print(output)
tups_dict = output[0].keys()

mat = [[0 for _ in range(9)] for _ in range(9)]
for x_str, y_str in tups_dict:
    x, y = int(x_str[1:]), int(y_str[1:])
    # print(x,y)
    mat[x-1][y-1] = output[0][(x_str, y_str)]


# print(mat)
for row in mat:
    print(row)

print(output[1])



## solving??
bqm = model.to_bqm()
sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=10)
decoded_samples = model.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)
print(best_sample.sample)  # output seems to be wrong w/ bruteforce, correct w/ einsum (but wrong matrix output?)