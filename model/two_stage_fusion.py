"""
@author:haoran
time:0329

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary
import numpy as np

class TwoStageFusion(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,s1,s2):


