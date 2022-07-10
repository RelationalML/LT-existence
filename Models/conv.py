import torch
import torch.nn as nn
import numpy as np
from Models import layers
from torch.nn import functional as F

def sigg(input):
    '''
    Applies the approximate sigmoid to the first layer parameters
    '''
    input[input==0] = 1 
    return input

class SiG(nn.Module):
    '''
    Applies the Approx. Sigmoid function element-wise:
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return sigg(input) # simply apply already implemented SiLU
    
class id_act(nn.Module):
    '''
    Applies the Approx. Sigmoid function element-wise:
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input

act_func = SiG()

identity = id_act()

def fc(input_shape, num_classes, dense_classifier=False, pretrained=False, L=4, N=500, nonlinearity=nn.Sigmoid()): #L=6, N=100, nonlinearity=nn.ReLU()):
  size = np.prod(input_shape)
  
  # Linear feature extractor
  modules = [nn.Flatten()]
  modules.append(layers.Linear(size, N))
  modules.append(nonlinearity)
  for i in range(L-2):
    modules.append(layers.Linear(N,N))
    modules.append(nonlinearity)

  # Linear classifier
  if dense_classifier:
    modules.append(nn.Linear(N, num_classes))
  else:
    modules.append(layers.Linear(N, num_classes))
  model = nn.Sequential(*modules)

  # Pretrained model
  if pretrained:
    print("WARNING: this model does not have pretrained weights.")
  
  return model


def conv(input_shape, num_classes, dense_classifier=False, pretrained=False, L=3, N=32, nonlinearity=nn.Sigmoid()): #nn.LeakyReLU()): 
  channels, width, height = input_shape
  
  # Convolutional feature extractor
  modules = []
  modules.append(layers.Conv2d(channels, N, kernel_size=3, padding=3//2))
  modules.append(nonlinearity)
  for i in range(L-2):
    modules.append(layers.Conv2d(N, N, kernel_size=3, padding=3//2))
    modules.append(nonlinearity)
      
  # Linear classifier
  modules.append(nn.Flatten())
  if dense_classifier:
    modules.append(nn.Linear(N * width * height, num_classes))
  else:
    modules.append(layers.Linear(N * width * height, num_classes))
  model = nn.Sequential(*modules)

  # Pretrained model
  if pretrained:
    print("WARNING: this model does not have pretrained weights.")
  
  return model


def conv2(input_shape, num_classes, wpruned, bpruned, nonlinearity=nn.Sigmoid(), non_first=act_func): 
  channels, width, height = input_shape
  
  # Convolutional feature extractor
  modules = []
  L = len(wpruned)
  pl = np.zeros(L)
  for l in range(L):
    pl[l] = wpruned[l].shape[0]
  modules.append(layers.Conv2d(channels, int(pl[0]), kernel_size=1, padding=0))
  modules.append(non_first)
  modules.append(layers.Conv2d(int(pl[0]), int(pl[1]), kernel_size=3, padding=3//2))
  modules.append(nonlinearity)
  
  for i in range(2,L-2,2):
    modules.append(layers.Conv2d(int(pl[i-1]), int(pl[i]), kernel_size=1, padding=0))
    modules.append(non_first)
    modules.append(layers.Conv2d(int(pl[i]), int(pl[i+1]), kernel_size=3, padding=3//2))
    modules.append(nonlinearity)
      
  # Linear classifier
  modules.append(nn.Flatten())
  modules.append(layers.Linear(wpruned[L-1].shape[1], num_classes))
  model = nn.Sequential(*modules)
  
  return model



def conv3(input_shape, num_classes, wpruned, bpruned, nonlinearity=nn.Sigmoid(), non_first=act_func): 
  channels, width, height = input_shape
  
  # Convolutional feature extractor
  modules = []
  L = len(wpruned)
  pl = np.zeros(L)
  for l in range(L):
    pl[l] = wpruned[l].shape[0]
  
  modules.append(layers.Conv2d(channels, int(pl[0]), kernel_size=1, padding=0))
  modules.append(non_first)
  for i in range(1,L-1):
    modules.append(layers.Conv2d(int(pl[i-1]), int(pl[i]), kernel_size=3, padding=3//2))
    modules.append(nonlinearity)
      
  # Linear classifier
  modules.append(nn.Flatten())
  modules.append(layers.Linear(wpruned[L-1].shape[1], num_classes))
  model = nn.Sequential(*modules)
  
  return model
