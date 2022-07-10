import torch
import numpy as np
import torch.nn as nn
from Utils import pruning
from Utils import utils
import random
import argparse
import pathlib
import pickle
from Models import conv

def prune_plus_save(r, act, eps, model, out, construct):
    random.seed(r)
    L, width, wtl, btl, scale = target_net_syn(model)
    wpruned, bpruned, architect = prune_fc(wtl[:(L-1)], btl[:(L-1)], 30, 15, False, 0, eps)
    #avoid taking care of flatten operation, only verify conv layers
    wpruned.append(wtl[L-1])
    bpruned.append(btl[L-1])
    with open('ticket2_conv_relu_'+str(r) + "_" + str(eps), 'wb') as f:
        pickle.dump([wpruned, bpruned, architect], f)

def main():
    global args
    parser = argparse.ArgumentParser(description='Constructing convolutional lottery tickets (LTs) from target models.')
    parser.add_argument('--error', type=float, default=0.01, metavar='eps', help='Allowed approximation error for each target parameter (default=0.01).')
    parser.add_argument('--rep', type=int, default=5, metavar='nbrRep',
                        help='Number of independent repetitions of LT construction for a given target (default: 5).')
    parser.add_argument('--act', type=str, default='relu', help='Activation function (default=relu). Choose between: relu, lrelu, tanh, sigmoid.',
                        choices=['relu', 'lrelu', 'tanh', 'sigmoid'])
    parser.add_argument('--model', type=str, default="model.pt",
                        help='Path to target model.')
    parser.add_argument('--ssa_size', type=int, default=15, metavar='rho',
                        help='Size of base set for subset sum approximation (and thus multiplicity of neuron construction in LT).')
    parser.add_argument('--out', type=str, default="LT",
                        help='Filename where to dump the constructed LT.')
    parser.add_argument('--construct', type=str, default="L+1",
                        help='Construction method: L+1 or 2L.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default=1).')
    parser.add_argument('--data', type=str, default='mnist', help='Currently only tested option: mnist.')
    #parser.add_argument('--device', type=str, default='cuda', help='Choices: cuda or cpu.')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    #define relevant variables
    eps = args.error
    rho = args.ssa_size
    nbrRep = args.rep
    act = args.act
    device = "cuda"
    #device = args.device
    verbose = True
    args.out = "./LT/"+args.out
    
    #load data for model evaluation
    input_shape, num_classes = utils.dimension(args.data) # (1, 28, 28), 10
    loss = nn.CrossEntropyLoss() 
    dataload, dataset = utils.dataloader(args.data, 32, False, 4)
    
    #Target model:
    target="./Targets/"+args.model
    L, width, wtl, btl, scale = pruning.target_net_syn(target)
    #count number of target model parameters
    nbr = 0
    for l in range(len(wtl)-1):
        nbr = nbr + np.sum(np.abs(wtl[l]) > eps) + np.sum(np.abs(btl[l]) > eps) 
    print("Number of target parameters: " + str(nbr))
    
    print("Target performance: ")
    if act=="relu":
        model = conv.conv(input_shape, num_classes,nonlinearity=nn.ReLU())      
    elif act=="lrelu":
        model = conv.conv(input_shape, num_classes,nonlinearity=nn.LeakyReLU())      
    elif act=="tanh":
        model = conv.conv(input_shape, num_classes,nonlinearity=nn.Tanh())      
    elif act=="sigmoid":
        model = conv.conv(input_shape, num_classes,nonlinearity=nn.Sigmoid())      
    else:
        print("Activation function not implemented.") 
    model.load_state_dict(torch.load(target, map_location=torch.device(device)), strict=False)
    model.cuda()
    average_loss, accuracy1 = utils.eval(model, loss, dataload, device, verbose)
    print(accuracy1)
    
    #Pruning for and evaluating LTs
    res = torch.zeros(nbrRep)
    param = torch.zeros(nbrRep)
    #L+1 construction
    if args.construct=="L+1":
        for r in range(nbrRep):
            if act=="relu":
                wpruned, bpruned, architect = pruning.prune_conv(wtl[:(L-1)], btl[:(L-1)], 2*rho, rho, False, 0, eps)
            elif act=="lrelu":
                wpruned, bpruned, architect = pruning.prune_conv(wtl[:(L-1)], btl[:(L-1)], 2*rho, rho, False, 0, eps)
            elif act=="tanh":
                wpruned, bpruned, architect = pruning.prune_conv(wtl[:(L-1)], btl[:(L-1)], rho, rho, True, 0, eps)
            elif act=="sigmoid":
                wpruned, bpruned, architect = pruning.prune_conv(wtl[:(L-1)], btl[:(L-1)], rho, rho, True, 1, eps)
            else:
                print("Activation function not implemented.")
            #avoid taking care of tedious flattening operation, only verify conv layers
            wpruned.append(wtl[L-1])
            bpruned.append(btl[L-1])
            with open(args.out+'_'+act+'_'+str(r)+"_"+str(eps), 'wb') as f:
                pickle.dump([wpruned, bpruned, architect], f)
            
            #evaluation
            if act=="relu":
                model = conv.conv3(input_shape, num_classes, wpruned, bpruned, nn.ReLU(), nn.ReLU())
            elif act=="lrelu":
                model = conv.conv3(input_shape, num_classes, wpruned, bpruned, nn.LeakyReLU(), nn.LeakyReLU())
            elif act=="tanh":
                model = conv.conv3(input_shape, num_classes, wpruned, bpruned, nn.Tanh(), mlp.identity)
            elif act=="sigmoid":
                model = conv.conv3(input_shape, num_classes, wpruned, bpruned)
            else:
                print("Activation function not implemented.")
                
            for l in range(len(wpruned)-1):
                param[r] = param[r] + np.sum(np.abs(wpruned[l]) > eps) + np.sum(np.abs(bpruned[l]) > eps) 
                
            model_dict = model.state_dict()
            model_dict2 = model_dict
            i=0
            for ll in model_dict2.keys():
                if ll.endswith("weight"):
                    model_dict2[ll].data = torch.tensor(wpruned[i], dtype=torch.float, device=torch.device('cuda')) 
                if ll.endswith("bias"):
                    model_dict2[ll].data = torch.tensor(bpruned[i], dtype=torch.float, device=torch.device('cuda')) 
                    i = i+1
            model_dict.update(model_dict2)
            model.load_state_dict(model_dict)
            model.cuda()
            average_loss, accuracy1 = utils.eval(model, loss, dataload, device, verbose)
            res[r] = accuracy1 
    
    else:
        #2L construction
        for r in range(nbrRep):
            if act=="relu":
                wpruned, bpruned, architect = pruning.prune_conv_2L(wtl[:(L-1)], btl[:(L-1)], 2*rho, False, 0, eps)
            elif act=="lrelu":
                wpruned, bpruned, architect = pruning.prune_conv_2L(wtl[:(L-1)], btl[:(L-1)], 2*rho, False, 0, eps)
            elif act=="tanh":
                wpruned, bpruned, architect = pruning.prune_conv_2L(wtl[:(L-1)], btl[:(L-1)], rho, True, 0, eps)
            elif act=="sigmoid":
                wpruned, bpruned, architect = pruning.prune_conv_2L(wtl[:(L-1)], btl[:(L-1)], rho, True, 1, eps)
            else:
                print("Activation function not implemented.")
            #avoid taking care of flatten operation, only verify conv layers
            wpruned.append(wtl[L-1])
            bpruned.append(btl[L-1])
            with open(args.out+'_2L_'+act+'_'+str(r)+"_"+str(eps), 'wb') as f:
                pickle.dump([wpruned, bpruned, architect], f)
    
            #evaluation
            if act=="relu":
                model = conv.conv2(input_shape, num_classes, wpruned, bpruned, nn.ReLU(), nn.ReLU())
            elif act=="lrelu":
                model = conv.conv2(input_shape, num_classes, wpruned, bpruned, nn.LeakyReLU(), nn.LeakyReLU())
            elif act=="tanh":
                model = conv.conv2(input_shape, num_classes, wpruned, bpruned, nn.Tanh(), mlp.identity)
            elif act=="sigmoid":
                model = conv.conv2(input_shape, num_classes, wpruned, bpruned)
            else:
                print("Activation function not implemented.")
                
            for l in range(len(wpruned)-1):
                param[r] = param[r] + np.sum(np.abs(wpruned[l]) > eps) + np.sum(np.abs(bpruned[l]) > eps) 
                
            model_dict = model.state_dict()
            model_dict2 = model_dict
            i=0
            for ll in model_dict2.keys():
                if ll.endswith("weight"):
                    model_dict2[ll].data = torch.tensor(wpruned[i], dtype=torch.float, device=torch.device('cuda')) 
                if ll.endswith("bias"):
                    model_dict2[ll].data = torch.tensor(bpruned[i], dtype=torch.float, device=torch.device('cuda')) 
                    i = i+1
            model_dict.update(model_dict2)
            model.load_state_dict(model_dict)
            model.cuda()
            average_loss, accuracy1 = utils.eval(model, loss, dataload, device, verbose)
            res[r] = accuracy1 
    
    print("Stats tickets:")
    print([torch.mean(res), torch.std(res)*1.96/np.sqrt(nbrRep), torch.mean(param), torch.std(param)*1.96/np.sqrt(nbrRep)])
    
if __name__ == '__main__':
    main()

            
