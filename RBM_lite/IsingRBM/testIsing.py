import sys
import os
import torch
import numpy as np
# homedir = '../'
# sys.path.insert(0,  homedir)
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

def testProb(W, b, samps=1000, trials=10, temperature=0.5, coupling=10, device='cpu', ising=False):
    model = IsingRBM(W=W, b=b, temperature=temperature, coupling=coupling, ising=ising)
    #If device is cuda, move to cuda
    model.to(device)
    energies = np.array(testProb_model(model, samps, trials))
    return energies