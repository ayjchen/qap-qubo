import sys
import torch
import numpy as np
import os
#So that RBM can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import RBM.utils as utils
from IsingRBM.IsingRBM import IsingRBM


def testProb_model(model, samps, trials):
    outStats = model.tensgenerate_statistics(samps, trials)[0]
    outCuts = []
    for i, sampDict in enumerate(outStats):
        v = list(sampDict.items())
        vals = [x[1] for x in v]
        MLE = utils.fromBuffer(v[np.argmax(vals)][0]).numpy()
        outCuts.append(model.ising_energy(MLE))
    return outCuts

def testProb(W=None, b=None, fname=None, samps=1000, trials=10, temperature=0.5, coupling=10, device='cpu', ising=False):
    if fname:
        model = IsingRBM(fname=fname, temperature=temperature, coupling=coupling, ising=ising)
    else:
        model = IsingRBM(W=W, b=b, temperature=temperature, coupling=coupling, ising=ising)
    #If device is cuda, move to cuda
    model.to(device)
    energies = np.array(testProb_model(model, samps, trials))
    return energies

if __name__ == "__main__":
    testFile = "../N010-id00.txt"
    model = IsingRBM(fname=testFile, temperature=0.5, coupling=10, ising=True)
    print("Weights", model.weights)
    print("Visible Biases", model.visible_bias)
    print("Hidden Biases", model.hidden_bias)
    testW = model.adj.numpy()
    testb = model.adj_b.numpy()
    model2 = IsingRBM(W=testW, b=testb, temperature=0.5, coupling=10, ising=True)
    print("Equal weights?", model2.weights == model.weights)
    print("Equal Visible Biases?", (model2.visible_bias == model.visible_bias))
    print("Equal Hidden Biases?", (model2.hidden_bias == model2.visible_bias))
    

    