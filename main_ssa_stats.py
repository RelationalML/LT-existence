import numpy as np
import matplotlib.pylab as plt
from itertools import combinations
import argparse
import random
import pickle

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

def expC(n, eps, samples, subset):
    rep = samples
    err = np.zeros(rep)
    ii = np.zeros(rep)
    for i in range(rep):
        target = np.random.uniform(-1,1,1)
        cand, ind = exhaustive(target, np.random.uniform(-1,1,n), eps, subset)
        err[i] = np.abs(cand-target)
        ii[i] = len(ind)
    delta = np.sum(err > eps)/rep
    if delta > 0:
        C = n/(-np.log(min(eps, delta)))
    else:
        C = 1
    return C, delta, err, ii

def main():
    global args
    parser = argparse.ArgumentParser(description='Constructing convolutional lottery tickets (LTs) from target models.')
    parser.add_argument('--error', type=float, default=0.01, metavar='eps', help='Allowed approximation error for each target parameter (default=0.01).')
    parser.add_argument('--rep', type=int, default=100000, metavar='nbrRep',
                        help='Number of independent repetitions of LT construction for a given target (default: 5).')
    parser.add_argument('--ssa_size', type=int, default=15, metavar='rho',
                        help='Size of base set for subset sum approximation (and thus multiplicity of neuron construction in LT).')
    parser.add_argument('--construct', type=str, default="L+1",
                        help='Construction method: L+1 or 2L.')
    parser.add_argument('--sub', type=int, default=3,
                        help='Maximum considered subset size.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default=1).')
    args = parser.parse_args()
    
    random.seed(args.seed)

    rep = args.rep
    err = np.zeros(rep)
    ii = np.zeros(rep)
    n = args.ssa_size
    eps = args.error
    
    if args.construct == "L+1":
        for i in range(rep):
            target = np.random.uniform(-1,1,1)
            cand, ind = exhaustive(target, np.random.uniform(-1,1,n), eps, args.sub) #exhaustiveBest(target, np.random.uniform(-1,1,n), 15)
            err[i] = np.abs(cand-target)
            ii[i] = len(ind)
            if ind[0] == (-1):
                ii[i] = 0
            if (i%1000 == 0) and (i>1):
                print(i)
                print(np.mean(err[(i-1000):i]))
        print("Mean error:")
        print(np.mean(err))
        print("Max error:")
        print(np.max(err))
        with open('./Subset_sum_stats/subset_stats_n_'+str(n)+'_sub_'+str(args.sub)+'_eps_'+str(eps), 'wb') as f:
            pickle.dump([err,ii], f)
    else:
        for i in range(rep):
            target = np.random.uniform(-1,1,1)
            cand, ind = exhaustive(target, np.random.uniform(-1,1,n)*np.random.uniform(-1,1,n), eps, args.sub)
            err[i] = np.abs(cand-target)
            ii[i] = len(ind)
            if ind[0] == (-1):
                ii[i] = 0
            if (i%1000 == 0) and (i>1):
                print(i)
                print(np.mean(err[(i-1000):i]))
        print("Mean error:")
        print(np.mean(err))
        print("Max error:")
        print(np.max(err))
        with open('./Subset_sum_stats/subset_2l_stats_n_'+str(n)+'_sub_'+str(args.sub)+'_eps_'+str(eps), 'wb') as f:
            pickle.dump([err,ii], f)

if __name__ == '__main__':
    main()
    
    
