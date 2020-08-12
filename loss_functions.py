import tensorflow as tf
from keras import backend as K
from keras.losses import mean_absolute_error
import numpy as np

import tensorflow.contrib.slim as slim
import math

def cal_base(y_true, y_pred):
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = K.round(K.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = K.sum(y_positive * y_pred_positive)
    TN = K.sum(y_negative * y_pred_negative)

    FP = K.sum(y_negative * y_pred_positive)
    FN = K.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN


def acc(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + K.epsilon())
    return ACC
def precision(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    PC = TP/(TP + FP + K.epsilon())
    return PC


def sensitivity(y_true, y_pred):
    """ recall """
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SE = TP/(TP + FN + K.epsilon())
    return SE





def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + K.epsilon())
    return SP


def f1_socre(y_true, y_pred):
    SE = sensitivity(y_true, y_pred)
    PC = precision(y_true, y_pred)
    F1 = 2 * SE * PC / (SE + PC + K.epsilon())
    return F1



def huber_loss(y_true,y_pred):
    return tf.losses.huber_loss(labels=y_true,predictions=y_pred)

def Weight_Mse_loss(y_true,y_pred):
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)

    y_pred = tf.log(y_pred / (1 - y_pred))
    y_true = tf.cast(y_true, tf.float32)
    y_weight =y_true
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)
    y_weight=pos_weight*y_weight
    cost=K.mean(y_weight*K.square(y_true-y_pred))
    return cost
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


def SSIM(img1, img2, k1=0.01, k2=0.02, L=1, window_size=11):
    """
    The function is to calculate the ssim score
    """

    # img1 = tf.expand_dims(img1, 0)
    # img1 = tf.expand_dims(img1, -1)
    # img2 = tf.expand_dims(img2, 0)
    # img2 = tf.expand_dims(img2, -1)
    window = _tf_fspecial_gauss(window_size)

    # _epsilon = _to_tensor(K.epsilon(), img2.dtype.base_dtype)
    # y_pred = tf.clip_by_value(img2, _epsilon, 1 - _epsilon)
    # y_pred = tf.log(y_pred / (1 - y_pred))
    # y_true = tf.cast(img1, tf.float32)


    mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')


    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides = [1 ,1, 1, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

    c1 = (k1*L)**2
    c2 = (k2*L)**2

    ssim_map = 1-((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))

    return tf.reduce_mean(ssim_map)

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


if __name__ == '__main__':
    pass