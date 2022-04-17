## data parsing
def parse_in(filename):

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


## solution parsing
def parse_out(filename):

    with open(filename) as file:
        data = file.read().split('\n')

    data = list(filter(lambda x : x.strip(), data))
    # print(data)

    data_iter = iter(data)

    n, objective = next(data_iter).split()
    n = int(n)
    objective = int(objective)
    pos = [k for k in list(map(int, next(data_iter).split()))]

    return objective, pos