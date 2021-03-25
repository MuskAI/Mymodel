"""
@author :haoran
time:0316
tolerance 指标
"""
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

class ToleranceMetrics:
    def __init__(self):
        pass

    def f1_score(self, pred, gt):
        label = gt.long()
        mask = (label != 0).float()
        num_positive = np.sum(mask).astype('float')
        num_negative = mask.numel() - num_positive
        # print (num_positive, num_negative)
        mask[mask != 0] = num_negative / (num_positive + num_negative)  # 0.995
        mask[mask == 0] = num_positive / (num_positive + num_negative)  # 0.005

        y = pred.reshape(-1)
        l = gt.reshape(-1)

        y = np.where(y > 0.5, 1, 0).astype('int')
        l = np.array(l.cpu().detach()).astype('int')

        return f1_score(y, l, zero_division=1)

        pass

    def value_f1_score(self, pred, label):
        band = self.__gen_band(Image.fromarray(label))
        gt = label
        _gt = np.where((gt == 255) | (gt == 100), 1, 0)
        area = np.where((gt == 255) | (gt == 50), 1, 0)
        label = _gt
        plt.imshow(band)
        plt.show()

        mask = (label != 0)
        num_positive = np.sum(mask).astype('float')
        num_negative = mask - num_positive
        # print (num_positive, num_negative)
        w1 = num_negative / (num_positive + num_negative)
        w2 = num_positive / (num_positive + num_negative)
        mask = np.where(mask != 0, w1, w2)

        y = pred.reshape(-1)
        l = label.reshape(-1)
        band = band.reshape(-1)
        area = area.reshape(-1)
        mask = mask.reshape(-1)
        y = np.where(y > 0.5, 1, 0).astype('int')
        l = np.array(l).astype('int')
        tn, fp, fn, tp = confusion_matrix(y_pred=y, y_true=l).ravel()

        factor_a = (tn + fp + fn + tp) / (2 * (fp + tn))
        factor_b = 1 * (tn + fp + fn + tp) / (2 * (tp + fn))
        print('The factor a,b:', factor_a, factor_b)
        print('tp, fn, fp, tn:', tp, fn, fp, tn)

        fp = fp * factor_a
        tn = tn * factor_a

        tp = tp * factor_b
        fn = fn * factor_b
        print('value tp, fn, fp, tn:', tp, fn, fp, tn)
        # tp: true positive，剿灭敌人的数量
        # fn: flase negative, 漏掉敌人的数量
        # fp: 误杀平民的数量
        # 假设一场战争的价值为1，平均到每一个人升上

        # weight compute

        # F1 = 2*(Precision*Recall)/(Precision+Recall)
        # Precision = TP/(TP+FP)
        # RECALL = TP/(TP+FN)
        acc = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print('The acc precision recall f1:', acc, precision, recall, f1)
        print('The acc:', accuracy_score(y_true=l, y_pred=y))
        print('The f1:', f1_score(y_true=l, y_pred=y))
        print('The recall:', recall_score(y_true=l, y_pred=y))
        print('The precision:', precision_score(y_true=l, y_pred=y))

    def __gen_band(self, gt, dilate_window=5):
        """

        :param gt: PIL type
        :param dilate_window:
        :return:
        """

        _gt = gt.copy()

        # input required
        if len(_gt.split()) == 3:
            _gt = _gt.split()[0]
        else:
            pass

        _gt = np.array(_gt, dtype='uint8')

        if max(_gt.reshape(-1)) == 255:
            _gt = np.where((_gt == 255) | (_gt == 100), 1, 0)
            _gt = np.array(_gt, dtype='uint8')
        else:
            pass

        _gt = cv.merge([_gt])
        kernel = np.ones((dilate_window, dilate_window), np.uint8)
        _band = cv.dilate(_gt, kernel)
        _band = np.array(_band, dtype='uint8')
        # _band = np.where(_band == 1, 255, 0)
        _band = Image.fromarray(np.array(_band, dtype='uint8'))
        if len(_band.split()) == 3:
            _band = np.array(_band)[:, :, 0]
        else:
            _band = np.array(_band)
        return _band
