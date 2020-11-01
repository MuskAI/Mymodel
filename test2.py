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
            # name = 'Default_78_398895_clock.png'
            # gt_name = name.replace('Default','Gt')
            # gt_name = gt_name.replace('png','bmp')
            # gt_name = gt_name.replace('jpg','bmp')
            # gt = Image.open(os.path.join(test_data_path,gt_name))
            # gt = np.array(gt)
            # gt = to_none_class_map(gt)
            # plt.figure('gt')
            # plt.imshow(gt)
            # plt.show()

            image_path = os.path.join(test_data_path,name)
            gt_path = Helper().find_gt(image_path)

            gt = Image.open(gt_path)
            gt = np.array(gt)
            plt.figure('gt')
            plt.imshow(gt)
            plt.show()
            gt = np.where((gt == 255) | (gt == 100), 1, 0)
            plt.figure('gt2')
            plt.imshow(gt)
            plt.show()
            gt_ = np.array(gt,dtype='float32')

            img = Image.open(image_path)
            img = np.array(img,dtype='float32')
            R_MEAN = img[:,:,0].mean()
            G_MEAN = img[:,:,1].mean()
            B_MEAN = img[:,:,2].mean()
            img[:,:,0] =img[:,:,0]-R_MEAN
            img[:, :, 1] = img[:, :, 1] - G_MEAN
            img[:, :, 2] = img[:, :, 2] - B_MEAN
            img[:, :, 0] /= 255
            img[:, :, 1] /= 255
            img[:, :, 2] /= 255

            img = np.transpose(img,(2,0,1))
            img = img[np.newaxis,:,:,:]
            img = torch.from_numpy(img)
            img = img.cpu()
            print(img.shape)
            output = model(img)
            output = output[0]

            output = np.array(output.cpu().detach().numpy(), dtype='float32')
            output = output.squeeze(0)



            output = np.transpose(output,(1,2,0))
            output_ = output.squeeze(2)

            # 在这里计算一下loss
            output_ = torch.from_numpy(output_)
            gt_ = torch.from_numpy(gt_)
            loss = torch.nn.functional.binary_cross_entropy(output_,gt_)
            print(loss)
            plt.figure('prediction')
            plt.imshow(output_)
            plt.show()
            output = np.array(output_)*255
            output = np.asarray(output,dtype='uint8')
            # output_img = Image.fromarray(output)
            # output_img.save(output_path + 'output_'+name)
            cv.imwrite(output_path + 'output_'+name,output)
    except Exception as e:
        traceback.print_exc()
        print(e)

class Helper():
    def __init__(self,test_src_dir = '/media/liu/File/少量调试数据2/debug_src',
                 test_gt_dir ='/media/liu/File/少量调试数据2/debug_gt'):
        self.test_src_dir = test_src_dir
        self.test_gt_dir = test_gt_dir
        pass
    def find_gt(self, src_path):
        """
        using src name to find gt
        using this function to validation loss
        using this funciton when debug
        :return: gt path
        """
        src_name = src_path.split('/')[-1]
        gt_name = src_name.replace('Default','Gt').replace('png','bmp').replace('jpg','bmp')
        gt_path = os.path.join(self.test_gt_dir,gt_name)
        if os.path.exists(gt_path):
            pass
        else:
            print(gt_path,'not exists')
            traceback.print_exc()
            sys.exit()
        return gt_path

if __name__ == '__main__':
    try:
        test_data_path = '/media/liu/File/少量调试数据2/debug_src'
        output_path = 'test_record/test_1030_3/'
        model_path = '/home/liu/chenhaoran/Mymodel/record823/1030checkpoint_epoch31.pth'
        checkpoint = torch.load(model_path)
        model = Net().cpu()
        # model = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        read_test_data()
    except Exception as e:
        traceback.print_exc()
        print(e)
