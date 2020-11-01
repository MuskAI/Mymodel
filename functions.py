import numpy as np
import torch
from sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score

# loss function

def sigmoid_cross_entropy_loss(prediction, label):
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


def cross_entropy_loss(prediction, label):
    label = label.long()
    mask = (label != 0).float()
    _ = np.array(mask.cpu())

    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    # print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative) # 0.995
    mask[mask == 0] = num_positive / (num_positive + num_negative) # 0.005
    _ = np.array(mask.cpu())
    # cost = torch.nn.functional.binary_cross_entropy(
    #         p,label.float(), weight=mask)
    cost = torch.nn.BCELoss()(prediction, label.float())
    # return torch.sum(cost)/(cost.size()[0]*cost.size()[1]*cost.size()[2]*cost.size()[3])
    return torch.sum(cost)
def weighted_nll_loss(prediction, label):
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = torch.nn.functional.nll_loss(prediction, label, reduce=False)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    return torch.sum(cost)/(cost.size()[0]*cost.size()[1]*cost.size()[2]*cost.size()[3])

def weighted_cross_entropy_loss(prediction, label, output_mask=False):
    criterion = torch.nn.CrossEntropyLoss(reduce=False)
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = criterion(prediction, label)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask == 1] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    if output_mask:
        return torch.sum(cost), (label != 0)
    else:
        return torch.sum(cost)

def l2_regression_loss(prediction, label, mask):
    label = torch.squeeze(label.float())
    prediction = torch.squeeze(prediction.float())
    mask = (mask != 0).float()
    num_positive = torch.sum(mask).float()
    cost = torch.nn.functional.mse_loss(prediction, label, reduce=False)
    cost = torch.mul(cost, mask)
    cost = cost / (num_positive + 0.00000001)
    return torch.sum(cost)

def l1_loss(prediction,label):
    return torch.nn.L1Loss()(prediction,label)


def smooth_l1_loss(prediction, label):
    return torch.nn.SmoothL1Loss()(prediction,label)

def CE_loss(prediction,label):
    cost = torch.nn.functional.binary_cross_entropy(prediction,label)
    return torch.sum(cost)


def debug_ce(prediction,label):
    cost = torch.nn.functional.binary_cross_entropy(prediction,label)
    return cost

def BCE_loss(prediction,label):
    loss1 = cross_entropy_loss(prediction,label)



def wce_huber_loss(prediction,label):
    loss1 = cross_entropy_loss(prediction,label)
    loss2 = smooth_l1_loss(prediction,label)
    loss3 = l1_loss(prediction,label)
    return 0.6*loss1+0.35*loss2+0.05*loss3


def wce_huber_loss_8(prediction,label):
    loss1 = cross_entropy_loss(prediction,label)
    loss2 = smooth_l1_loss(prediction,label)
    return 0.6*loss1+0.4*loss2
def my_precision_score(prediction,label):
    y = prediction.reshape(prediction.size()[0]*prediction.size()[1]*prediction.size()[2]*prediction.size()[3])
    l = label.reshape(label.size()[0] * label.size()[1] * label.size()[2] * label.size()[3])
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0)
    l = np.array(l.cpu().detach())
    return precision_score(y, l)

def my_acc_score(prediction,label):
    y = prediction.reshape(prediction.size()[0]*prediction.size()[1]*prediction.size()[2]*prediction.size()[3])
    l = label.reshape(label.size()[0] * label.size()[1] * label.size()[2] * label.size()[3])
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0)
    l = np.array(l.cpu().detach())
    return accuracy_score(y,l)

def my_f1_score(prediction,label):

    y = prediction.reshape(prediction.size()[0]*prediction.size()[1]*prediction.size()[2]*prediction.size()[3])
    l = label.reshape(label.size()[0] * label.size()[1] * label.size()[2] * label.size()[3])

    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')

    return f1_score(y,l)

def my_recall_score(prediction,label):

    y = prediction.reshape(prediction.size()[0]*prediction.size()[1]*prediction.size()[2]*prediction.size()[3])
    l = label.reshape(label.size()[0] * label.size()[1] * label.size()[2] * label.size()[3])

    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')

    return recall_score(y,l)
