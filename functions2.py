import torch
import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score
import torch.nn.functional as F


def bce2d(input, target):
    n, c, h, w = input.size()
    # assert(max(target) == 1)
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t > 0)
    neg_index = (target_t == 0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
    return loss

if __name__ == '__main__':
    a = np.random.randn(2,3,320,320)
    b = np.random.randn(2,3,320,320)
    a = np.where(a<0,0,1)
    b = np.where(a < 0, 0, 1)
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    bce2d(a,b)