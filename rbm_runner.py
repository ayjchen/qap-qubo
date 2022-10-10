from cmath import exp
from pyqubo import Binary
import numpy as np
import neal
import sys
from joblib import Parallel, delayed
import logging

import input_parser as prs
sys.path.append('/home/achen/qap-qubo/RBM_lite/IsingRBM')
import testIsing as ising

# def main_func(reads, sweeps):
filename = "nug12"



M = 20000000.0  # penalty coeff

##### start of process #####
# n, f, d = prs.parse_in("qapdata/" + filename + ".dat")
n, f, d = prs.parse_in("example.in")
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
    p_terms += (sum([x[j+i*n] for j in range(n)]) - 1)**2 + (sum([x[j*n+i] for j in range(n)]) - 1)**2
p_terms *= M
H += p_terms


## compiling qubo model ###########
model = H.compile()
output = model.to_qubo()
tups_dict = output[0].keys()
# print("Penalty additive constant:", output[1])

### qubo matrix ###
mat = [[0 for _ in range(n**2)] for _ in range(n**2)]
for x_str, y_str in tups_dict:
    x, y = int(x_str[1:]), int(y_str[1:])
    # print(x,y)
    mat[x][y] = output[0][(x_str, y_str)]

bias = []
for i in range(n**2):
    bias.append(mat[i][i])
    mat[i][i] = 0

# print(mat)
# for row in mat:
#     print(row)
# print(bias)
### END qubo matrix ###

testProb_output = ising.testProb(mat, bias)

print(testProb_output)



#     ## solving ############
#     bqm = model.to_bqm()
#     print(type(model))
#     # neal
#     sa = neal.SimulatedAnnealingSampler()
#     sampleset = sa.sample(bqm, num_reads=reads, num_sweeps=sweeps, beta_range=(10**(-2), 10**(1)))
#     decoded_samples = model.decode_sampleset(sampleset)
#     best_sample = min(decoded_samples, key=lambda x: x.energy)
#     best = best_sample.sample

#     ## converting output to matrix
#     best_keys = best.keys()
#     x_mat = [[0 for _ in range(n)] for _ in range(n)]
#     for x_str in best_keys:
#         x = int(x_str[1:])
#         val = int(best[x_str])
#         row = x // n
#         col = x % n
#         x_mat[row][col] = val

#     # # debug print assignment matrix
#     # pos = []
#     # for i in range(n):
#     #     print(x_mat[i])
#     #     pos.append(x_mat[i].index(1)+1)




#     ## getting objective function val
#     x = x_mat
#     obj = 0

#     for i in range(n):
#         for j in range(n):
#             for k in range(n):
#                 for l in range(n):
#                     obj += f[i][j]*d[k][l]*x[i][k]*x[j][l]

#     expected_obj, expected_pos = prs.parse_out("qapsoln/" + filename + ".sln")

#     # print("Expected positions:", expected_pos)
#     # print("Actual positions:", pos)
#     # print("Expected opt obj:", expected_obj, "\nActual output obj:", obj)
#     return obj


# def run_func(reads=10, sweeps=1000):
#     # arr = []
#     # for i in range(100):
#     #     arr.append(main_func(reads, sweeps))
#     # return arr
#     return Parallel(n_jobs=16)(delayed(main_func)(reads, sweeps) for i in range(500))

# def var_reads(start, end, step):
#     for num_reads in range(start, end, step):
#         log_msg = "var_reads=" + str(num_reads)
#         logging.info(log_msg)
#         print(num_reads, '=', run_func(reads=num_reads))

# def var_sweeps(start, end, step):
#     for num_sweeps in range(start, end, step):
#         log_msg = "var_sweeps=" + str(num_sweeps)
#         logging.info(log_msg)
#         print(num_sweeps, '=', run_func(sweeps=num_sweeps))


# def main(argv):
#     for _ in range(10):
#         start, end, step = argv
#         start, end, step = int(start), int(end), int(step)
#         output_filename = "avg_time_" + str(start) + ".log"
#         logging.basicConfig(filename=output_filename, format='%(asctime)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)

#         var_reads(start, end, step)
#         logging.info("Process completed")

# if __name__ == '__main__':
#     main(sys.argv[1:])