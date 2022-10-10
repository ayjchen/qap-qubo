import numpy as np
import torch
import logging
from RBM.rbm import *


def AIS(model, temps, num_runs):
    """
    Peforms an Annealed Importance Sampling Run
    See: Salakhutidinov and Murray, 2008 "On the Quantitative Analysis of Deep Belief Networks"
    args: model, temps, num_runs
    model - the model we want the partition function of
    temps - inverse temperatures to be used for sampling (should be a list of values from 0 to 1)
            the final value of temps should be 1 to make sure we end with the correct model
    num_runs - number of parallel runs to perform at once

    returns: log_Z
    log_Z - the log of the partition function of the input model
    """

    zero_rbm = RBM(model.num_visible, model.num_hidden, 1)
    zero_rbm.visible_bias = torch.tensor(model.visible_bias)
    zero_rbm.hidden_bias = torch.zeros_like(model.hidden_bias)
    zero_rbm.weights = torch.zeros_like(model.weights)

    v = zero_rbm.tensgenerate_sample(num_runs, use_outbits=False)

    pk0 = zero_rbm
    pk1 = RBM(model.num_visible, model.num_hidden, 1)

    #Setting to zeros for numerical stability
    w = torch.zeros(num_runs)
    #
    for beta in temps:
        pk1.weights =  beta * model.weights

        pk1.visible_bias = model.visible_bias

        pk1.hidden_bias = beta * model.hidden_bias

        #Using logs of probabilities to keep things numerically stable
        log_prob1 = pk1.prob(v, partition=False, log=True)
        log_prob2 = pk0.prob(v, partition=False, log=True)



        #This calcualtes the log of the importance weights
        w += (log_prob1 - log_prob2)

        #Generating next set of data vectors
        v = pk1.tensgenerate_sample(num_runs, input_data=v, use_outbits=False)

        pk0 = pk1

    log_Za = torch.log(torch.Tensor([2])) * zero_rbm.num_hidden \
            + torch.sum(torch.log1p(torch.exp(zero_rbm.visible_bias)))
    #print(log_Za)
    log_Z = torch.logsumexp(w, 0) - torch.log(torch.Tensor([num_runs])) + log_Za
    return log_Z


def AnnealedMCMC(model, temps, num_runs, clamp=None, use_outbits=True, debug=False):
    temps = torch.tensor(temps).to(torch.device(model.device))

    if clamp is None:
        clamp = torch.zeros(model.num_visible, device=torch.device(model.device)) - 1
    else:
        temp = clamp
        clamp = torch.zeros(model.num_visible, device=torch.device(model.device)) - 1
        clamp[model.outbits] = temp
        clamp[model.zeros] = 0

    zero_rbm = RBM(model.num_visible, model.num_hidden, 1, use_cuda=model.use_cuda)
    zero_rbm.cuda(device=model.device)
    zero_rbm.visible_bias = model.visible_bias
    #zero_rbm.visible_bias = torch.zeros_like(model.visible_bias)
    zero_rbm.hidden_bias = torch.zeros_like(model.hidden_bias)
    zero_rbm.weights = torch.zeros_like(model.weights)

    v = zero_rbm.tensgenerate_sample(num_runs, clamp=clamp, use_zeros=True, use_outbits=False)


    #Setting to zeros for numerical stability
    pk1 = RBM(model.num_visible, model.num_hidden, 1, use_cuda=model.use_cuda)
    pk1.cuda(device=model.device)
    for i, beta in enumerate(temps):
        if i%5000 == 0 and debug:
            logging.info(beta)
            out = {}
            for val in v.cpu():
                #Gets the raw bytes and converts them to a string to use as key
                #This is significantly faster than any other method.
                key = val.numpy().tostring()

                if key in out:
                    out[key][1] += 1
                else:
                    out[key] = [val, 1]


            samps = list(out.values())
            vals = [x[1] for x in samps]
            tens = [x[0] for x in samps]
            sort = np.flip(np.argsort(vals), 0)
            maxes = [samps[sort[i]] for i in range(len(samps))]

            for x in maxes[:5]:
                logging.info(utils.MultToString([y.item() for y in x[0][model.outbits]]) + ', ' + str(x[1]))



        pk1.weights = beta * model.weights

        pk1.visible_bias = model.visible_bias

        pk1.hidden_bias = beta * model.hidden_bias

        v = pk1.tensgenerate_sample(num_runs, input_data=v, clamp=clamp, use_zeros=True, use_outbits=False)

    if use_outbits:
        return v[:, model.outbits]

    return v
