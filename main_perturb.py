import torch
import torch.nn as nn
import numpy as np
#from Utils pruning import *
import random
import argparse
import pathlib
import pickle
from os.path import exists
from Models import lottery_resnet
from Utils import utils

def main():
    global args
    parser = argparse.ArgumentParser(description='Estimating the approximation error of a LT (obtained by an L+1 construction) that approximates a given target network and sufficient number of parameters of the corresponding source network.')
    parser.add_argument('--error', type=float, default=0.01, metavar='eps', help='Allowed approximation error for each target parameter (default=0.01).')
    parser.add_argument('--rep', type=int, default=50, metavar='nbrRep',
                        help='Number of independent repetitions of LT construction for a given target (default: 5).')
    parser.add_argument('--act', type=str, default='sigmoid', help='Activation function (default=sigmoid). Choose between: relu, lrelu, tanh, sigmoid.',
                        choices=['relu', 'lrelu', 'tanh', 'sigmoid'])
    parser.add_argument('--model', type=str, default="resnet.pt",
                        help='Path to target model.')
    parser.add_argument('--ssa_size', type=int, default=15, metavar='rho',
                        help='Size of base set for subset sum approximation (and thus multiplicity of neuron construction in LT).')
    parser.add_argument('--data', type=str, default='cifar10', help='Currently only tested option: cifar10.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default=1).')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    #define relevant variables
    eps = args.error
    rho = args.ssa_size
    nbrRep = args.rep
    act = args.act
    
    #statistics of solving independent subset sum approximation problems
    path_stats = './Subset_sum_stats/subset_stats_n_'+str(rho)+'_sub_'+str(rho)+'_eps_'+str(eps)
    if exists(path_stats):
        with open(path_stats, 'rb') as f:
            err, subsetsize = pickle.load(f)
        ns = len(err)
    else:
        print("Generate statistics on solving independent subset sum approximation problems for the desired case.")

    #2L construction for first layer 
    path_stats = './Subset_sum_stats/subset_2l_stats_n_'+str(rho)+'_sub_'+str(rho)+'_eps_'+str(eps)
    if exists(path_stats):
        with open(path_stats, 'rb') as f:
            err2, subsetsize2 = pickle.load(f)
        ns2 = len(err2)
    else:
        print("Generate statistics on solving independent subset sum approximation problems based on products of random weights for the desired case.")
    #note: generate those files if they are not already available
    
    #take account of unacceptable error in estimation of required number of parameters
    delta = np.sum(err > eps)/ns
    delta2 = np.sum(err2 > eps)/ns2

    subsetsize = subsetsize[err <= eps]
    err = err[err <= eps]
    subsetsize2 = subsetsize2[err2 <= eps]
    err2 = err2[err2 <= eps]
    ns = len(err)
    ns2 = len(err2)
    err = torch.tensor(err).cuda()
    err2 = torch.tensor(err2).cuda()
    subsetsize = torch.tensor(subsetsize).cuda()
    subsetsize2 = torch.tensor(subsetsize2).cuda()
    
    #define perturbations of a target network as function to call it in multiple independent realizations
    def approx_target(path):
        target_dict =  torch.load(path, map_location=torch.device('cuda')) 
        nbr_params_target = torch.tensor([0]).cuda()
        nbr_params = torch.tensor([4*15]).cuda()
        layer = 0
        for ll in target_dict.keys():
            target_dict[ll].data = torch.tensor(target_dict[ll], dtype=torch.float, device=torch.device('cuda')) 
            x = ll.split(".")
            if (("conv1" in x) or ("conv2" in x) or ("conv" in x) or ("fc" in x)): 
                if ll.endswith("weight"):
                    mask = target_dict[ll+'_mask'].data 
                    wt = target_dict[ll].data 
                    wt = wt*mask
                    if layer == 0:
                        mm = torch.randint(0, ns2, wt.size()).cuda()
                        error = err2[mm]*(-1)**torch.randint(0,2,wt.size()).cuda()  
                        npp = subsetsize2[mm]
                        fac = 2 
                    else:
                        mm = torch.randint(0, ns, wt.size()).cuda()
                        error = err[mm]*(-1)**torch.randint(0,2,wt.size()).cuda()
                        npp = subsetsize[mm]
                        fac = rho 
                    wt = (torch.abs(wt) > eps) * error + wt
                    nbr_params = nbr_params + torch.sum(npp[torch.abs(wt)>eps])*fac
                    nbr_params_target = nbr_params_target + torch.sum(torch.abs(wt)>eps)
                    target_dict[ll].data = wt.cuda()
                    nbNew = torch.sum(npp[torch.abs(wt)>eps])*fac
                    layer = layer+1
                if ll.endswith("bias"):
                    mask = target_dict[ll+'_mask'].data
                    bt = target_dict[ll].data
                    bt = bt*mask
                    if layer == 0:
                        mm = torch.randint(0, ns2, bt.size()).cuda()
                        error = err2[mm]*(-1)**torch.randint(0,2,bt.size()).cuda()
                        npp = subsetsize2[mm]
                        fac = rho 
                    else:
                        mm = torch.randint(0, ns, bt.shape).cuda()
                        error = err[mm]*(-1)**torch.randint(0,2,bt.size()).cuda()
                        npp = subsetsize[mm]
                        fac = rho 
                    bt = (torch.abs(bt) > eps) * error + bt
                    nbr_params = nbr_params + torch.sum(npp[torch.abs(bt)>eps])*fac
                    nbr_params_target = nbr_params_target + torch.sum(torch.abs(bt)>eps)
                    target_dict[ll].data = bt.cuda()
                    layer = layer+1
                    nbNew = nbNew + torch.sum(npp[torch.abs(bt)>eps])*fac
        #do not need to create rho neurons for each target neuron in last layer
        nbr_params = nbr_params - nbNew*(rho-1)/rho
        print(nbr_params_target)        
        return target_dict, nbr_params, nbr_params_target
    
    
    #load test data
    input_shape, num_classes = utils.dimension(args.data)
    dataload, dataset = utils.dataloader(args.data, 32, False, 4)
    device = "cuda"
    verbose = True
    loss = nn.CrossEntropyLoss()
    #define model 
    if act=="relu":
        model = lottery_resnet.resnet20(input_shape, num_classes, nn.ReLU(), False, False) 
    elif act=="lrelu":
        model = lottery_resnet.resnet20(input_shape, num_classes, nn.LeakyReLU(), False, False) 
    elif act=="tanh":
        model = lottery_resnet.resnet20(input_shape, num_classes, nn.Tanh(), False, False) 
    elif act=="sigmoid":
        model = lottery_resnet.resnet20(input_shape, num_classes, nn.Sigmoid(), False, False) 
    else:
        print("Activation function not implemented.") 
        

    acc = torch.zeros(nbrRep)
    npp = torch.zeros(nbrRep)
    target = "./Targets/"+args.model
    for i in range(nbrRep):
        print("Model " + str(i))
        target_dict, nbr_params, nbr_params_target = approx_target(target)
        model.load_state_dict(target_dict, strict=False)
        model.cuda()
        average_loss, accuracy1 = utils.eval(model, loss, dataload, device, verbose)
        acc[i] = accuracy1
        npp[i] = nbr_params

    print("LT stats:")
    print([torch.mean(acc), torch.std(acc)*1.96/np.sqrt(nbrRep), torch.mean(npp), torch.std(npp)*1.96/np.sqrt(nbrRep)])
    sorted_acc, ind = torch.sort(acc)
    print("Top 1/2 of LTs:")
    print([torch.mean(sorted_acc[int(nbrRep/2):]), torch.std(sorted_acc[int(nbrRep/2):])*1.96/np.sqrt(nbrRep/2), torch.mean(npp[ind[int(nbrRep/2):]]), torch.std(npp[ind[int(nbrRep/2):]])*1.96/np.sqrt(nbrRep/2)])

    print("Target: ")
    model.load_state_dict(torch.load(target, map_location=torch.device('cuda')), strict=False)
    model.cuda()
    average_loss, accuracy1 = utils.eval(model, loss, dataload, device, verbose)


if __name__ == '__main__':
    main()        
            
