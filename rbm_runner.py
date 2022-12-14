from cmath import exp
from pyqubo import Binary
import numpy as np
import neal
import sys
from joblib import Parallel, delayed
from collections import Counter
from itertools import groupby as g
import logging
import heapq as hq
import cProfile

import input_parser as prs
sys.path.append('/home/achen/qap-qubo/RBM_lite/IsingRBM')
import testIsing as ising
from IsingRBM import IsingRBM

# def main_func(reads, sweeps):
filename = "nug12"


def run_func(penalty, coup, temp):
    # M = 1000.0  # penalty coeff
    M = float(penalty)

    ##### start of process #####
    # n, f, d = prs.parse_in("qapdata/" + filename + ".dat")
    n, f, d = prs.parse_in("small_examples/3x3.in")
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
        val = output[0][(x_str, y_str)] / 2
        mat[x][y] = val
        mat[y][x] = val

    bias = []
    for i in range(n**2):
        bias.append(mat[i][i])
        mat[i][i] = 0
    # bias = [0] * (n**2)

    # print(mat)
    # print("mat=")
    # for row in mat:
    #     print(row)
    # print("bias=", bias)
    ### END qubo matrix ###

    testProb_output, raw_samples = ising.testProb(W=mat, b=bias, samps=5000, trials=100, temperature=temp, coupling=coup, ising=True, rawSamples=True)    
    testProb_output = list(map(lambda x : tuple(x), testProb_output))
    print(raw_samples)
    most_probable_output = Counter(testProb_output).most_common(1)[0][0]
    # print("penalty=", penalty, "coup=", coup, "temp=", temp)
    # print(most_probable_output)
    ########### smaller temp, larger coupling

    # model = IsingRBM.IsingRBM(W=mat, b=bias, temperature=0.001, coupling=coup, ising=True)
    # testProb_output = model.probs(log=True)

    # print("penalty=", penalty, "coup=", coup)
    
    # testProb_output = testProb_output.numpy()
    # max_prob = max(testProb_output)
    # # print(testProb_output)
    # max_prob = hq.nlargest(10, testProb_output)
    # print("maxprob=", max_prob)
    # imax = np.argwhere(testProb_output == np.amax(testProb_output))
    # imin = np.argmin(testProb_output)
    # print("max=", imax.flatten().tolist(), "min=", imin)
    # # print("prob of ground state=", testProb_output[272])

    ## converting output to matrix
    x_mat = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(len(most_probable_output)):
        x_mat[i//n][i%n] = most_probable_output[i]
    print(x_mat)

    expected_output = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    num_correct = testProb_output.count(expected_output)
    print("penalty=", penalty, "coup=", coup, "temp=", temp, "num_correct=", num_correct)

# for i in range(100, 1000, 100):
#     run_func(penalty=i, coup=i*2, temp=0.1)
# for i in range(1000, 10000, 1000):
#     run_func(penalty=i, coup=i*2, temp=0.001)

# for i in range(1, 10):
#     run_func(penalty=1000, coup=2000, temp=i*0.0001)
# for i in range(1, 10):
#     run_func(penalty=1000, coup=2000, temp=i*0.001)

run_func(penalty=1000, coup=2000, temp=0.001)

# for i in range (1000, 10000, 1000):
#     for j in range (i, 10000, 1000):
#         run_func(penalty=i, coup=j)

# run_func(penalty=100, coup=10000, temp=0.001)
# run_func(penalty=10000, coup=100, temp=0.001)
# run_func(penalty=10000, coup=10000, temp=0.001)
# run_func(penalty=100, coup=100, temp=0.001)
# run_func(penalty=100,scoup=10000, temp=0.1)
# run_func(penalty=10000, coup=100, temp=0.1)
# run_func(penalty=10000, coup=10000, temp=0.1)
# run_func(penalty=100, coup=100, temp=0.1)

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