import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score
import torch.nn.functional as F

#
# class Evaluation:
#     def __init__(self,output,label):
#         self.output = output
#         self.label = label
#
#     def start_evaluation(self,compute_f1 = True,compute_precision= True,compute_accuracy = True,compute_recall = True):
#
#         if compute_f1:
#             pass
#         if compute_precision:
#             pass
#         if compute_recall:
#             pass
#         if compute_accuracy:
#             pass
#
#         return


def wce_dice_huber_loss(pred, gt):
    loss1 = cross_entropy_loss(pred, gt)
    loss2 = DiceLoss()(pred, gt)
    loss3 = smooth_l1_loss(pred, gt)
    return 0.6 * loss1 + 0.3 * loss2 + 0.1 * loss3



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


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
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask)
    # cost = torch.nn.BCELoss()(prediction, label.float())

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
    y = prediction.reshape(-1)
    l = label.reshape(-1)
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')
    return precision_score(y, l, zero_division=1)

def my_acc_score(prediction,label):
    y = prediction.reshape(-1)
    l = label.reshape(-1)
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')
    return accuracy_score(y,l)

def my_f1_score(prediction,label):

    y = prediction.reshape(-1)
    l = label.reshape(-1)

    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')

    return f1_score(y,l,zero_division=1)

def my_recall_score(prediction,label):

    y = prediction.reshape(-1)
    l = label.reshape(-1)
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')

    return recall_score(y, l, zero_division=1)

if __name__ == '__main__':
    a = torch.randn(1,1,320,320)
    b = torch.randn(1, 1, 320, 320)
    a = nn.Sigmoid()(a)
    b = nn.Sigmoid()(b)
    loss = wce_dice_huber_loss(a,b)
    print(loss)