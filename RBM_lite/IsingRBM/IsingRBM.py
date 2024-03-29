import torch
import sys
import os
import numpy as np
#So that RBM can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RBM.rbm import RBM
import RBM.utils as utils
from numba import jit

class IsingRBM(RBM):
    '''
    RBM for evaluating Ising Model (fully connected {-1, 1} activation}  problems
    Parameters:
    fname - Filename (see sample N010-id00.txt) which contains file in "ising" format (assumes {-1, 1} activations)
    W - if not using fname parameter, can directly give weight matrix. Format can be numpy, pytorch, or regular python
    b - if not using fname parameter, biases for weight matrix
    temperature - inverse temperatre (beta) parameter that scales the weight matrix for slower or faster mixing time
    coupling - strength of connection between copies of the same node. 
    ising - Whether to have the input remain in {-1, 1} format (ising=True) or move the input to {0, 1} format (ising=False)
        If no modifications to input weights needed, set ising=True
    '''
    def __init__(self, fname=None, W=None, b=None, temperature=1, coupling=10, ising=True):
        '''
        fname is file to MaxCUT data from
        coupling is the graph embedding coupling parameter
        '''
        self.temperature = temperature
        self.coupling = coupling
        #Ising formulation ({-1, 1} activation) vs Boltzmann formulation ({0, 1} activation)
        self.ising = ising
        if fname:
            f = open(fname, 'r')
            #First Line is parameters, number of vertices and number of edges
            params = f.readline()[:-1].split()
            params = list(map(float, params))
            params = list(map(int, params))
            super().__init__(params[0], params[0], 1)
            self.num_vertices = params[0]
            self.num_edges = params[1]
            self.weights = torch.zeros_like(self.weights)
            self.visible_bias = torch.zeros_like(self.visible_bias)
            self.hidden_bias = torch.zeros_like(self.hidden_bias)

            #Adjacency matrix
            self.adj = torch.zeros(params[0], params[0])

            for line in f:
                inds = line[:-1].split()
                #First two indices
                inds[0] = int(float(inds[0]))
                inds[1] = int(float(inds[1]))
                #Weight value between them (can be float or int)
                inds[2] = float(inds[2])
                self.weights[inds[0]-1, inds[1]-1] = inds[2]
                self.weights[inds[1]-1, inds[0]-1] = inds[2]
                #keeping track of the adjacency matrix before we do modifications on it
                self.adj[inds[0]-1, inds[1] - 1] = inds[2]
                self.adj[inds[1]-1, inds[0] - 1] = inds[2]
            self.adj_b = torch.zeros_like(self.visible_bias)
            f.close()
        else:
            self.num_vertices = len(b)
            super().__init__(self.num_vertices, self.num_vertices, 1)
            for i, line in enumerate(W):
                # print("i=", i, "line=", line)
                for j, val in enumerate(line):
                    self.weights[i, j] = float(val)
                    self.weights[j, i] = float(val)
                    # print("j=", j, "val=", val)
            for i, val in enumerate(b):
                self.visible_bias[i] = float(val)
                self.hidden_bias[i] = float(val)
            self.adj = torch.Tensor(W)
            self.adj_b = torch.Tensor(b)
            # print(self.weights)
            
        #Adding coupling term
        for i in range(self.num_visible):
            self.weights[i, i] = -1 * coupling

        #This inverts the weight matrix (so that cuts are incentivized)
        self.weights = -1 * self.weights
        self.visible_bias = -1 * self.visible_bias
        self.hidden_bias = -1 * self.hidden_bias
        # print("self.weights=", self.weights)

        if self.ising:
            self.visible_bias = temperature * self.visible_bias
            self.hidden_bias = temperature * self.hidden_bias
            self.weights = temperature * self.weights
        else:
        #Going from a s = {-1, 1} to s = {0, 1}
            self.visible_bias = temperature * 2*(self.visible_bias - torch.matmul(self.weights.t(), torch.ones(self.num_visible)))
            self.hidden_bias = temperature * 2*(self.hidden_bias - torch.matmul(self.weights, torch.ones(self.num_hidden)))
            self.weights = temperature * 4*self.weights

        
    """
    Calculates the energy of a given state. Assumes input adjacency matrix is in ising format. 
    if ising=True, modifies the state to use ising weight matrix assumptions ({-1, 1} activations)
    if ising=False, no modification to state ({0, 1} activations)
    """
    def ising_energy(self, state, ising=True):
        adj = self.adj.cpu().numpy()
        adj_b = self.adj_b.cpu().numpy()
        if type(state) == torch.Tensor:
            state = state.cpu().numpy()
        if ising:
            state2 = 2 * state - 1
        else:
            state2 = state
        return np.matmul(np.matmul(state2, adj), state2) + np.dot(state2, adj_b)



class CutRBM(IsingRBM):
    def cut_value(self, state):
        '''
        Takes a state and returns the value of the cut created by
        partitioning through those states
        '''
        '''
        cut = 0
        for i in range(self.num_visible):
            outgoing_edges = self.adj[i, :]
            for j in range(i):
                if outgoing_edges[j] != 0:
                    #states are on different side of cut
                    if state[i] != state[j]:
                        cut += 1
        '''
        adj = self.adj.cpu().numpy()
        if type(state) == torch.Tensor:
            state = state.cpu().numpy()
        return CutRBM.cut_value_jit(adj, self.num_visible, state)
    @jit(nopython=True, cache=True)
    def cut_value_jit(adj, num_visible, state):
        cut = 0
        for i in range(num_visible):
            outgoing_edges = adj[i, :]
            for j in range(i):
                if outgoing_edges[j] != 0:
                    if state[i] != state[j]:
                        cut += adj[i, j]
        return cut
