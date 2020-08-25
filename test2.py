"""
created by haoran
time 8-17
"""
import traceback
from model import model
from model.model_812 import Net
import os, sys
import numpy as np
from PIL import Image
import shutil
import argparse
import time
import datetime
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2 as cv
from functions import sigmoid_cross_entropy_loss, cross_entropy_loss,l1_loss,wce_huber_loss
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from utils import to_none_class_map

def read_test_data():
    try:
        image_name = os.listdir(test_data_path)
        for name in image_name:
            print(image_name)
            name = 'Default_78_398895_clock.png'
            gt_name = name.replace('Default','Gt')
            gt_name = gt_name.replace('png','bmp')
            gt_name = gt_name.replace('jpg','bmp')
            gt = Image.open(gt_name)
            gt = np.array(gt)
            gt = to_none_class_map(gt)
            plt.figure('gt')
            plt.imshow(gt)
            plt.show()

            image_path = os.path.join(test_data_path,name)
            img = Image.open('Default_78_398895_clock.png')
            img = np.array(img,dtype='float32')
            R_MEAN = img[:,:,0].mean()
            G_MEAN = img[:,:,1].mean()
            B_MEAN = img[:,:,2].mean()
            img[:,:,0] =img[:,:,0]-R_MEAN
            img[:, :, 1] = img[:, :, 1] - G_MEAN
            img[:, :, 2] = img[:, :, 2] - B_MEAN

            img = np.transpose(img,(2,0,1))
            img = img[np.newaxis,:,:,:]
            img = torch.from_numpy(img)
            img = img.cuda()
            print(img.shape)
            output = model(img)
            output = output[0]


            gt = np.array(gt,dtype='float32')
            gt = gt[np.newaxis, :, :]

            gt = gt[np.newaxis, :, :, :]
            gt = torch.from_numpy(gt).cuda()

            loss = wce_huber_loss(output, gt)
            print(loss)
            output = np.array(output.cpu().detach().numpy(), dtype='float32')
            output = output.squeeze(0)



            output = np.transpose(output,(1,2,0))
            output_ = output.squeeze(2)
            plt.figure('prediction')
            plt.imshow(output)
            plt.show()
            output = np.array(output)*255
            output = np.asarray(output,dtype='uint8')
            # output_img = Image.fromarray(output)
            # output_img.save(output_path + 'output_'+name)
            cv.imwrite(output_path + 'output_'+name,output)
    except Exception as e:
        traceback.print_exc()
        print(e)




if __name__ == '__main__':
    try:
        test_data_path = '/home/liu/chenhaoran/Mymodel_wkl/mid_result_821_val/mid_result_epoch_0/mid_label'
        output_path = './'
        model_path = '/home/liu/chenhaoran/Mymodel_wkl/record823/epoch-1-checkpoint.pth'
        checkpoint = torch.load(model_path)
        model = Net().cuda()
        # model = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        read_test_data()
    except Exception as e:
        traceback.print_exc()
        print(e)
