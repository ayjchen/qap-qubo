import subprocess
import sys

script_name = 'index.py'
output_prefix = 'out'
# n_iter = 5

# for i in range(n_iter):
#     output_file = output_prefix + '_' + str(i) + '.out'
#     sys.stdout = open(output_file, 'w')
#     subprocess.call(['python', script_name], stdout=sys.stdout, stderr=subprocess.STDOUT)


output_file = 'var_reads.out'
sys.stdout = open(output_file, 'w')
subprocess.call(['python', script_name], stdout=sys.stdout, stderr=subprocess.STDOUT)