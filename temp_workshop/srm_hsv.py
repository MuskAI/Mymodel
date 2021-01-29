"""
created by haoran
time:1128
noise map
light map
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image





def SRMLayer(x):
    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = np.asarray(
        [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])  # shape=(3,3,5,5)
    filters = np.transpose(filters, (2, 3, 1, 0))  # shape=(5,5,3,3)

    # initializer_srm = keras.initializers.Constant(filters)
    initializer_srm = torch.nn.init.constant(filters)
    # output = Conv2D(3, (5, 5), padding='same', kernel_initializer=initializer_srm, use_bias=False, trainable=False)(x)
    output = torch.nn.Conv2d(3,1,5,initializer_srm[0])(x)

    return output


if __name__ == '__main__':
    image_path = 'D:\\实验室\\1127splicing\\tamper_result\\Default_2623_46847_boat.jpg'
    I = Image.open(image_path)
    I = np.array(I,dtype='float32')
    I_tensor = torch.from_numpy(I)
    SRMLayer(I_tensor)