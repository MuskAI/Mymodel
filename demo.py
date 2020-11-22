import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from functions import my_precision_score,my_f1_score
if __name__ == '__main__':
    gt = np.zeros([320,320])
    pred = np.zeros([320,320])
    gt = torch.from_numpy(gt)
    pred = torch.from_numpy(pred)
    print(my_f1_score(pred,gt))
