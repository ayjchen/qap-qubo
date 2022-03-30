from pyqubo import Binary
import numpy as np

## input parsing
dim = input()
dim = int(dim)
f = []
d = []

# constructing input matrices
for i in range(0, dim):
    f.append([k for k in list(map(int, input().split()))])

for i in range(0, dim):
    d.append([k for k in list(map(int, input().split()))])

print(f, d)



x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18 = Binary('x1'), Binary('x2'), Binary('x3'), Binary('x4'), Binary('x5'), Binary('x6'), Binary('x7'), Binary('x8'), Binary('x9'), Binary('x10'), Binary('x11'), Binary('x12'), Binary('x13'), Binary('x14'), Binary('x15'), Binary('x16'), Binary('x17'), Binary('x18')
H = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) * (x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18)
print(H.compile().to_qubo())