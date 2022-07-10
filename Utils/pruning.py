import numpy as np
import matplotlib.pylab as plt
from itertools import combinations
import torch
import torch.nn as nn
import pickle

def relu(x):
    return np.clip(x, a_min=0, a_max=None)

def sigmoid_proxy(x):
    if np.abs(x) > 0:
        y=x
    else:
        y=1
    return y

sig = np.vectorize(sigmoid_proxy)

def net_eval(x, weight, bias):
    L = len(bias)
    for i in range(L-1):
        x = relu(weight[i]@x + bias[i])
    i = L-1
    x = weight[i]@x + bias[i]
    return x

def subset_fixed_size(target, numbers, eps, subsize, errBest):
    n = len(numbers)
    cand = 0
    indBest = np.array([np.NAN])
    for ind in combinations(range(n),subsize):
        inda = np.array(ind,dtype="int")
        napprox = np.sum(numbers[inda])
        diff = np.abs(target-napprox)
        if diff < errBest:
            errBest = diff
            cand = napprox
            indBest = inda
        if diff <= eps:
            break
    return cand, indBest, errBest

def exhaustive(target, numbers, eps, nmax):
    n = len(numbers)
    err = np.abs(target)
    errBest = err
    cand = 0
    indBest = np.array([-1])
    nmax = min(nmax, n)
    for k in range(nmax):
        cank, indk, errk = subset_fixed_size(target, numbers, eps, k, errBest)
        if errk < errBest:
            errBest = errk
            cand = cank
            indBest = indk
        if errBest <= eps:
            break
    return cand, indBest

def subset_fixed_size_best(target, numbers, subsize, errBest):
    n = len(numbers)
    cand = 0
    indBest = np.array([-1])
    for ind in combinations(range(n),subsize):
        inda = np.array(ind,dtype="int")
        napprox = np.sum(numbers[inda])
        diff = np.abs(target-napprox)
        if diff < errBest:
            errBest = diff
            cand = napprox
            indBest = inda
    if np.abs(target-cand) > errBest:
        errBest = np.abs(target)
    return cand, indBest, errBest

def exhaustiveBest(target, numbers, nmax):
    n = len(numbers)
    err = np.abs(target)
    errBest = err
    cand = 0
    indBest = np.array([-1])
    nmax = min(nmax, n)
    for k in range(nmax):
        cank, indk, errk = subset_fixed_size_best(target, numbers, k, errBest)
        if errk < errBest:
            errBest = errk
            cand = cank
            indBest = indk
    return cand, indBest
 
def solve_subset_sum(target, nbrVar, eps, addwidth):
    variables = np.random.uniform(-1,1,nbrVar)
    cand, ind = exhaustive(target, variables, eps, 10)
    #cand, ind = exhaustiveBest(target, variables, 10)
    err = np.abs(target-cand)
    subset = variables[ind]
    if err <= eps:
        return cand, subset, ind, addwidth
    else:
        addwidth = addwidth+1
        return solve_subset_sum(target, nbrVar, eps, addwidth)
    
def solve_subset_sum_first(target, nbrVar, eps, addwidth, wp1):
    variables = np.random.uniform(-1,1,nbrVar)
    cand, ind = exhaustive(target, variables*wp1, eps, 10)
    #cand, ind = exhaustiveBest(target, variables*wp1, 10)
    err = np.abs(target-cand)
    subset = variables[ind]
    if err <= eps:
        return cand, subset, ind, addwidth
    else:
        addwidth = addwidth+1
        return solve_subset_sum_first(target, nbrVar, eps, addwidth, wp1)
    
   
def parallel_weight_pruning(rho_in, rho_out, n_in, n_out, nrest, eps, wp, indIn, indOut, wtflat):
    addwidth = 0
    for i in range(n_in):
        indRange = np.arange(rho_in*i,rho_in*(i+1))
        for j in range(n_out):
            for l in range(nrest):
                target = wtflat[indOut[j],indIn[i],l]
                if(np.abs(target) > eps):
                    for k in range(rho_out):
                        param, subset, ind, nbrTrials = solve_subset_sum(target, rho_in, eps, 0)
                        wp[j*rho_out + k, indRange[ind], l] = subset
                        addwidth = addwidth+nbrTrials
    return wp, addwidth

def parallel_bias_pruning(rho_in, rho_out, n_in, n_out, nrest, eps, wp, indIn, indOut, bt, b_act):
    addwidth = 0
    if b_act == 0:
        for j in range(n_out):
            target = bt[indOut[j]]
            if(np.abs(target) > eps):
                for k in range(rho_out):
                    param, subset, ind, nbrTrials = solve_subset_sum(target, rho_in*nrest, eps, 0)
                    #nbrTrials = max(0,nbrTrials-nrest+1)
                    if nrest > 1:
                        for m in range(len(ind)):
                            i = ind[m]
                            l = i%nrest
                            ii = int((i-l)/nrest)
                            wp[j*rho_out + k, n_in*rho_in + ii, l] = subset[m]
                    else:
                        wp[j*rho_out + k, n_in*rho_in + ind, 0] = subset
                    addwidth = addwidth+nbrTrials
    else:
        for j in range(n_out):
            target = bt[indOut[j]]
            if(np.abs(target) > eps):
                for k in range(rho_out):
                    param, subset, ind, nbrTrials = solve_subset_sum_first(target, rho_in*nrest, eps, 0, b_act)
                    if nrest > 1:
                        for m in range(len(ind)):
                            i = ind[m]
                            l = i%nrest
                            ii = int((i-l)/nrest)
                            wp[j*rho_out + k, n_in*rho_in + ii, l] = subset[m]
                    else:
                        wp[j*rho_out + k, n_in*rho_in + ind, 0] = subset
                    addwidth = addwidth+nbrTrials
    return wp, addwidth  


def parallel_bias_pruning_flatten(ind_bias_start, rho_in, rho_out, n_in, n_out, nrest, eps, wp, indIn, indOut, bt, b_act):
    addwidth = 0
    if b_act == 0:
        for j in range(n_out):
            target = bt[indOut[j]]
            if(np.abs(target) > eps):
                for k in range(rho_out):
                    param, subset, ind, nbrTrials = solve_subset_sum(target, rho_in*nrest, eps, 0)
                    if nrest > 1:
                        for m in range(len(ind)):
                            i = ind[m]
                            l = i%nrest
                            ii = int((i-l)/nrest)
                            wp[j*rho_out + k, ind_bias_start + ii, l] = subset[m]
                    else:
                        wp[j*rho_out + k, ind_bias_start + ind, 0] = subset
                    addwidth = addwidth+nbrTrials
    else:
        for j in range(n_out):
            target = bt[indOut[j]]
            if(np.abs(target) > eps):
                for k in range(rho_out):
                    param, subset, ind, nbrTrials = solve_subset_sum_first(target, rho_in*nrest, eps, 0, b_act)
                    if nrest > 1:
                        for m in range(len(ind)):
                            i = ind[m]
                            l = i%nrest
                            ii = int((i-l)/nrest)
                            wp[j*rho_out + k, ind_bias_start + ii, l] = subset[m]
                    else:
                        wp[j*rho_out + k, ind_bias_start + ind, 0] = subset
                    addwidth = addwidth+nbrTrials
    return wp, addwidth 

def parallel_constant_pruning(rho_in, rho_out, n_in, n_out, nrest, eps, wp, indIn, indOut):
    target = 1
    addwidth = 0
    for k in range(rho_out):
        param, subset, ind, nbrTrials = solve_subset_sum(target, rho_in*nrest, eps, 0)
        if nrest > 1:
            for m in range(len(ind)):
                i = ind[m]
                l = i%nrest
                ii = int((i-l)/nrest)
                wp[n_out*rho_out + k, n_in*rho_in + ii, l] = subset[m]
        else:
            wp[n_out*rho_out + k, n_in*rho_in + ind, 0] = subset
        addwidth = addwidth+nbrTrials
    return wp, addwidth

#first layer
def parallel_weight_pruning_first(rho_in, rho_out, n_in, n_out, nrest, eps, wp1, wp2, indIn, indOut, wtflat, idact):
    addwidth = 0
    if idact:
        for i in range(n_in):
            indRange = np.arange(rho_in*i,rho_in*(i+1))
            for j in range(n_out):
                for l in range(nrest):
                    target = wtflat[indOut[j],indIn[i],l]
                    if(np.abs(target) > eps):
                        for k in range(rho_out):
                            param, subset, ind, nbrTrials = solve_subset_sum_first(target, rho_in, eps, 0, wp1[indRange,indIn[i],0])
                            wp2[j*rho_out + k, indRange[ind], l] = subset
                            addwidth = addwidth+nbrTrials
    else:
        for i in range(n_in):
            indRange = np.arange(rho_in*i,rho_in*(i+1))
            wcand = wp1[indRange,indIn[i],0].copy()
            indRangePos = indRange[wcand>0]
            indRangeNeg = indRange[wcand<0]
            wcandPos = wcand[wcand>0]
            wcandNeg = wcand[wcand<0]
            for j in range(n_out):
                for l in range(nrest):
                    target = wtflat[indOut[j],indIn[i],l]
                    if(np.abs(target) > eps):
                        for k in range(rho_out):
                            param, subset, ind, nbrTrials = solve_subset_sum_first(target, len(wcandPos), eps, 0, wcandPos)
                            wp2[j*rho_out + k, indRangePos[ind], l] = subset
                            addwidth = addwidth+nbrTrials
                            param, subset, ind, nbrTrials = solve_subset_sum_first(target, len(wcandNeg), eps, 0, wcandNeg)
                            wp2[j*rho_out + k, indRangeNeg[ind], l] = subset
                            addwidth = addwidth+nbrTrials
    return wp2, addwidth

def parallel_bias_pruning_first(rho_in, rho_out, n_in, n_out, nrest, eps, wp2, bp, indIn, indOut, bt, b_act):
    addwidth = 0
    bout = np.zeros(len(bp))
    if b_act == 0:
        indRange = np.arange(rho_in*n_in,rho_in*(n_in+1))
        bp1 = bp[indRange]
        for j in range(n_out):
            target = bt[indOut[j]]
            if(np.abs(target) > eps):
                for k in range(rho_out):
                    param, subset, ind, nbrTrials = solve_subset_sum_first(target, rho_in*nrest, eps, 0, np.repeat(bp1,nrest))
                    addwidth = addwidth+nbrTrials
                    if nrest > 1:
                        for m in range(len(ind)):
                            i = ind[m]
                            l = i%nrest
                            ii = int((i-l)/nrest)
                            wp2[j*rho_out + k, n_in*rho_in + ii, l] = subset[m]
                            bout[n_in*rho_in+ii] = bp1[ii]
                    else:
                        wp2[j*rho_out + k, n_in*rho_in + ind, 0] = subset
                        bout[n_in*rho_in+ind] = bp1[ind]
    else:
        for j in range(n_out):
            target = bt[indOut[j]]
            if(np.abs(target) > eps):
                for k in range(rho_out):
                    param, subset, ind, nbrTrials = solve_subset_sum_first(target, rho_in*nrest, eps, 0, b_act)
                    if nrest > 1:
                        for m in range(len(ind)):
                            i = ind[m]
                            l = i%nrest
                            ii = int((i-l)/nrest)
                            wp2[j*rho_out + k, n_in*rho_in + ii, l] = subset[m]
                    else:
                        wp2[j*rho_out + k, n_in*rho_in + ind, 0] = subset
                    addwidth = addwidth+nbrTrials
    return wp2, bout, addwidth  

def parallel_constant_pruning_first(rho_in, rho_out, n_in, n_out, nrest, eps, wp2, bp, indIn, indOut):
    target = 1
    addwidth = 0
    indRange = np.arange(rho_in*n_in,rho_in*(n_in+1))
    bp1 = bp[indRange]
    bp1 = np.repeat(bp1,nrest)
    bout = np.zeros(len(bp))
    for k in range(rho_out):
        param, subset, ind, nbrTrials = solve_subset_sum_first(target, rho_in*nrest, eps, 0, bp1)
        if nrest > 1:
            for m in range(len(ind)):
                i = ind[m]
                l = i%nrest
                ii = int((i-l)/nrest)
                wp2[n_out*rho_out + k, n_in*rho_in + ii, l] = subset[m]
                bout[n_in*rho_in + ii] = bp1[i]
        else:
            wp2[n_out*rho_out + k, n_in*rho_in + ind, 0] = subset
            bout[n_in*rho_in + ind] = bp1[ind]
        addwidth = addwidth+nbrTrials
    return wp2, bout, addwidth

def prune_layer(wt, bt, rho_in, rho_out, eps, LastLayer, b_act):
    # wt target tensor of dimension nt_out x nt_in: target weight parameters
    # rho_in: input multiplicity, i.e., number of available copies of input
    # rho_out output multiplicity, i.e., number of copies that should be created of every target output
    # eps allowed error in each parameter
    dimt = wt.shape #out_channels, groupsin_channels​, kernel_size[0],kernel_size[1])\text{kernel\_size[0]}, \text{kernel\_size[1]})kernel_size[0],kernel_size[1])
    n_out = dimt[0]
    n_in = dimt[1]
    wtflat = wt.copy().reshape(n_out, n_in, -1)
    nrest = wtflat.shape[2]
    #inputs in need of construction:
    degOut = np.sum(np.abs(wtflat) > eps, axis=(0,2))
    indIn = np.arange(n_in)
    indIn = indIn[degOut > 0]
    #ouputs in need of construction
    degIn = np.sum(np.abs(wtflat) > eps, axis=(1,2))
    indOut = np.arange(n_out)
    indOut = indOut[degIn > 0]
    n_out = len(indOut)
    n_in = len(indIn)
    ## needed in case of flattening later:
    n_out = dimt[0]
    n_in = dimt[1]
    indOut = np.arange(n_out)
    indIn = np.arange(n_in)
    ##
    dimp = tuple([rho_out*(n_out+1), rho_in*(n_in+1), nrest]) 
    #dimp[0] = rho_out*(n_out+1) #+1 for biases
    if LastLayer:
        rho_out = 1
        dimp = tuple([n_out, rho_in*(n_in+1), nrest])
    wp = np.zeros(dimp)
    wp, addwidth = parallel_weight_pruning(rho_in, rho_out, n_in, n_out, nrest, eps, wp, indIn, indOut, wtflat)
    #biases
    wp, addw = parallel_bias_pruning(rho_in, rho_out, n_in, n_out, nrest, eps, wp, indIn, indOut, bt, b_act)  
    addwidth = addwidth + addw
    #create rho_out bias neurons
    if ((LastLayer==False) and (b_act == 0)):
        wp, addw = parallel_constant_pruning(rho_in, rho_out, n_in, n_out, nrest, eps, wp, indIn, indOut)  
        addwidth = addwidth + addw
    wp = wp.reshape((dimp[0],dimp[1])+dimt[2:])
    return wp, addwidth


def prune_layer_first(wt, bt, rho_in, rho_out, b_act, eps, idact):
    #assume w_act scaling already taken care of by initialization
    #if idact=True: m+=m- so that there is no need to distinguish positive and negative inputs
    dimt = wt.shape #out_channels, in_channels​, kernel_size[0], kernel_size[1])
    n_out = dimt[0]
    n_in = dimt[1]
    wtflat = wt.copy().reshape(n_out, n_in, -1)
    nrest = wtflat.shape[2]
    #prune for weights
    degOut = np.sum(np.abs(wtflat) > eps, axis=(0,2))
    indIn = np.arange(n_in)
    indIn = indIn[degOut > 0]
    ##
    ## needed in case of flattening later:
    n_out = dimt[0]
    n_in = dimt[1]
    indOut = np.arange(n_out)
    indIn = np.arange(n_in)
    ##
    dimp2 = tuple([rho_out*(n_out+1), rho_in*(n_in+1), nrest]) #wtflat.shape #dim_t
    dimp1 = tuple([rho_in*(n_in+1), wt.shape[1], 1]) #tuple([rho_in*(n_in+1), wt.shape[1], nrest])
    #dimp[0] = rho_out*(n_out+1) #+1 for biases
    #parameters after pruning
    wp1 = np.zeros(dimp1)
    bp1 = np.zeros(rho_in*(n_in+1))
    #first layer init
    for i in range(n_in):
        indRange = np.arange(i*rho_in, (i+1)*rho_in)
        wp1[indRange,indIn[i],0] = np.random.uniform(-2,2,rho_in) # for simplicity, only stride = 1, other strides would maybe also require non-zero  wp1[indRange,indIn[i],1] etc.
    #bias
    indRange = np.arange(n_in*rho_in, (n_in+1)*rho_in)
    if b_act == 0:
        bp1[indRange] = np.random.uniform(0,2,rho_in) #for simplicity, these are aranged like that, with high probability we can first select biases, nodes with positive biases to identify the nodes that we want to prune down to biases
        
    wp2 = np.zeros(dimp2)
    #weight pruning
    wp2, addwidth = parallel_weight_pruning_first(rho_in, rho_out, n_in, n_out, nrest, eps, wp1, wp2, indIn, indOut, wtflat, idact)
    #biases
    wp2, bp1, addw = parallel_bias_pruning_first(rho_in, rho_out, n_in, n_out, nrest, eps, wp2, bp1, indIn, indOut, bt, b_act)  
    addwidth = addwidth + addw
    #create rho_out bias neurons
    if b_act == 0:
        wp2, bp1, addw = parallel_constant_pruning_first(rho_in, rho_out, n_in, n_out, nrest, eps, wp2, bp1, indIn, indOut)  
        addwidth = addwidth + addw
    if len(dimt) == 2:
        wp1 = wp1.reshape((dimp1[0],dimp1[1]))
        wp2 = wp2.reshape((dimp2[0],dimp2[1]))
    else:
        wp1 = wp1.reshape((dimp1[0],dimp1[1],1,1)) #+tuple(np.ones(len(dimt[2:]))))
        wp2 = wp2.reshape((dimp2[0],dimp2[1])+dimt[2:])
    return wp1, wp2, bp1, addwidth

def prune_layer_first_2L(wt, bt, rho_in, b_act, eps, idact):
    #assume w_act scaling already taken care of by initialization
    #if idact=True: m+=m_ so that there is no need to distinguish positive and negative inputs
    dimt = wt.shape #out_channels, in_channels​, kernel_size[0],kernel_size[1])
    n_out = dimt[0]
    n_in = dimt[1]
    wtflat = wt.copy().reshape(n_out, n_in, -1)
    nrest = wtflat.shape[2]
    #inputs in need of construction:
    degOut = np.sum(np.abs(wtflat) > eps, axis=(0,2))
    indIn = np.arange(n_in)
    indIn = indIn[degOut > 0]
    #ouputs in need of construction
    degIn = np.sum(np.abs(wtflat) > eps, axis=(1,2))
    indOut = np.arange(n_out)
    indOut = indOut[degIn > 0]
    n_out = len(indOut)
    n_in = len(indIn)
    ## needed in case of flattening later:
    n_out = dimt[0]
    n_in = dimt[1]
    indOut = np.arange(n_out)
    indIn = np.arange(n_in)
    ##
    #assume that 
    dimp2 = tuple([n_out, rho_in*(n_in+1), nrest]) #wtflat.shape #dim_t
    dimp1 = tuple([rho_in*(n_in+1), wt.shape[1], 1]) #tuple([rho_in*(n_in+1), wt.shape[1], nrest])
    #parameters after pruning
    wp1 = np.zeros(dimp1)
    bp1 = np.zeros(rho_in*(n_in+1))
    #first layer init
    for i in range(n_in):
        indRange = np.arange(i*rho_in, (i+1)*rho_in)
        wp1[indRange,indIn[i],0] = np.random.uniform(-2,2,rho_in) # for simplicity, only stride = 1, other strides would maybe also require non-zero  wp1[indRange,indIn[i],1] etc.
    #bias
    indRange = np.arange(n_in*rho_in, (n_in+1)*rho_in)
    if b_act == 0:
        bp1[indRange] = np.random.uniform(0,2,rho_in) #for simplicity, these are aranged like that, with high probability we can first select biases nodes with positive biases to identify the nodes that we want to prune down to biases
        
    wp2 = np.zeros(dimp2)
    #weight pruning
    wp2, addwidth = parallel_weight_pruning_first(rho_in, 1, n_in, n_out, nrest, eps, wp1, wp2, indIn, indOut, wtflat, idact)
    #biases
    wp2, bp1, addw = parallel_bias_pruning_first(rho_in, 1, n_in, n_out, nrest, eps, wp2, bp1, indIn, indOut, bt, b_act)  
    addwidth = addwidth + addw
    if len(dimt) == 2:
        wp1 = wp1.reshape((dimp1[0],dimp1[1]))
        wp2 = wp2.reshape((dimp2[0],dimp2[1]))
    else:
        wp1 = wp1.reshape((dimp1[0],dimp1[1],1,1))
        wp2 = wp2.reshape((dimp2[0],dimp2[1])+dimt[2:])
    return wp1, wp2, bp1, addwidth


def prune_conv(wt, bt, rho_in_1, rho, idact, b_act, eps):
    #L+1 construction
    #List of target network weights: wt
    #List of target network biases: bt
    #Multiplicity of neuron construction in first layer: rho_in_1 (If the target layer consists of nt neurons/channels, the LT consistis of rho_in_1*nt neurons/channels
    #Multiplicity of neuron construction in the other layers: rho 
    #If idact=True: m+=m_ so that there is no need to distinguish positive and negative inputs
    #Intercept of activation function: b_act 
    #Allowed error per paramter: eps
    L = len(wt)
    wpruned = list()
    bpruned = list()
    architect = np.zeros(L+2)
    #start in first layer
    l=0
    print("l: " + str(l))
    wp1, wp2, bp1, addwidth = prune_layer_first(wt[l], bt[l], rho_in_1, rho, b_act, eps, idact)
    wpruned.append(wp1)
    wpruned.append(wp2)
    bpruned.append(bp1)
    bpruned.append(np.zeros(wp2.shape[0]))
    architect[0] = wp1.shape[1]
    architect[1] = wp1.shape[0]
    architect[2] = wp2.shape[0] + addwidth
    LastLayer = False
    for l in range(1,L):
        if l < (L-1):
            print("l: " + str(l))
            wp, addwidth = prune_layer(wt[l], bt[l], rho, rho, eps, LastLayer, b_act)
        else:
            LastLayer = True
            print("l: " + str(l))
            wp, addwidth = prune_layer(wt[l], bt[l], rho, 1, eps, LastLayer, b_act)
        wpruned.append(wp)
        bpruned.append(np.zeros(wp.shape[0]))
        architect[l+2] = wp.shape[0] + addwidth
    return wpruned, bpruned, architect


def prune_conv_2L(wt, bt, rho, idact, b_act, eps):
    L = len(wt)
    wpruned = list()
    bpruned = list()
    architect = np.zeros(2*L+1)
    architect[0] = wt[0].shape[1]
    for l in range(L):
        print("l: " + str(l))
        wp1, wp2, bp1, addwidth = prune_layer_first_2L(wt[l], bt[l], rho, b_act, eps, idact)
        wpruned.append(wp1)
        wpruned.append(wp2)
        bpruned.append(bp1)
        bpruned.append(np.zeros(wp2.shape[0]))
        architect[2*l+1] = wp2.shape[1] 
        architect[2*l+2] = wp2.shape[0] + addwidth
    return wpruned, bpruned, architect


def target_net(pathTarget): 
    target_dict = torch.load(pathTarget)
    target_params = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), target_dict.items()))
    wtl = list()
    btl = list()
    L=0
    width=0
    scale = list()
    for ll in target_params.keys():
        if ll.endswith("weight"):
            wt = target_params[ll].data.clone().cpu().detach().numpy()
            wtl.append(wt)
            sc = np.max(np.abs(wtl[L]))
        if ll.endswith("bias"):
            bt = target_params[ll].data.clone().cpu().detach().numpy()
            btl.append(bt)
            width = max(width,len(bt))
            sc = max(sc,np.max(np.abs(wtl[L])))
            scale.append(sc)
            L=L+1
    scale = np.array(scale)
    return L, width, wtl, btl, scale

def target_net_syn(pathTarget):
    target_dict = torch.load(pathTarget)
    target_params = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), target_dict.items()))
    wtl = list()
    btl = list()
    L=0
    width=0
    scale = list()
    for ll in target_params.keys():
        if ll.endswith("weight"):
            mask = target_dict[ll+'_mask'].data.clone().cpu().detach().numpy()
            wt = target_params[ll].data.clone().cpu().detach().numpy()
            wtl.append(wt*mask)
            sc = np.max(np.abs(wtl[L]))
        if ll.endswith("bias"):
            mask = target_dict[ll+'_mask'].data.clone().cpu().detach().numpy()
            bt = target_params[ll].data.clone().cpu().detach().numpy()
            btl.append(bt*mask)
            width = max(width,len(bt))
            sc = max(sc,np.max(np.abs(wtl[L])))
            scale.append(sc)
            L=L+1
    scale = np.array(scale)
    return L, width, wtl, btl, scale

def number_params(weight, bias, eps):
    L = len(weight)
    nn=0
    for l in range(L):
        nn = nn + np.sum(np.abs(weight[l]) >= eps)
        print(str(l) + ": " + str(np.sum(np.abs(weight[l]) >= eps))) 
    L = len(bias)
    for l in range(L):
        nn = nn + np.sum(np.abs(bias[l]) >= eps)
    return nn
            
