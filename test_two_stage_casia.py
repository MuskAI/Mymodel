"""
created by haoran
time 8-17
"""
import traceback
from model import model
from model.model_two_stage import Net_Stage_1 as Net1
from model.model_two_stage import Net_Stage_2 as Net2

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
from functions import sigmoid_cross_entropy_loss, cross_entropy_loss, l1_loss, wce_huber_loss
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from utils import to_none_class_map

device = torch.device("cpu")


def read_test_data(output_path):
    try:
        image_name = os.listdir(test_data_path)
        length = len(image_name)
        for index, name in enumerate(image_name):
            print(index, '/', length)
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

            image_path = os.path.join(test_data_path, name)
            # gt_path = Helper().find_gt(image_path)

            # gt = Image.open(gt_path)
            # gt = np.array(gt)
            # plt.figure('gt')
            # plt.imshow(gt)
            # plt.show()
            # gt = np.where((gt == 255) | (gt == 100), 1, 0)
            # plt.figure('gt2')
            # plt.imshow(gt)
            # plt.show()
            # gt_ = np.array(gt,dtype='float32')

            img = Image.open(image_path)
            img = np.array(img, dtype='float32')
            R_MEAN = img[:, :, 0].mean()
            G_MEAN = img[:, :, 1].mean()
            B_MEAN = img[:, :, 2].mean()
            img[:, :, 0] = img[:, :, 0] - R_MEAN
            img[:, :, 1] = img[:, :, 1] - G_MEAN
            img[:, :, 2] = img[:, :, 2] - B_MEAN
            img[:, :, 0] /= 255
            img[:, :, 1] /= 255
            img[:, :, 2] /= 255

            img = np.transpose(img, (2, 0, 1))
            img = img[np.newaxis, :, :, :]
            img = torch.from_numpy(img)
            img = img.cpu()
            # print(img.shape)
            output = model1(img)

            stage1_ouput = output[0]
            zero = torch.zeros_like(stage1_ouput)
            one = torch.ones_like(stage1_ouput)

            rgb_pred = img * torch.where(stage1_ouput > 0.1, one, zero)

            _rgb_pred = rgb_pred.squeeze(0)
            _rgb_pred = np.array(_rgb_pred)
            _rgb_pred = np.transpose(_rgb_pred, (1, 2, 0))
            # plt.imshow(_rgb_pred)
            # plt.show()

            model2_input = torch.cat((rgb_pred, img), 1)
            output2 = model2(model2_input, output[9], output[10], output[11])
            output = output[0]

            output = np.array(output.cpu().detach().numpy(), dtype='float32')
            output = output.squeeze(0)
            output = np.transpose(output, (1, 2, 0))
            output_ = output.squeeze(2)

            output2 = np.array(output2[0].cpu().detach().numpy(), dtype='float32')
            output2 = output2.squeeze(0)
            output2 = np.transpose(output2, (1, 2, 0))
            output2_ = output2.squeeze(2)
            # 在这里计算一下loss
            # output_ = torch.from_numpy(output_)
            # gt_ = torch.from_numpy(gt_)
            # loss = wce_huber_loss(output_,gt_)
            # print(loss)
            # plt.figure('prediction')
            # plt.imshow(output_)
            # plt.show()
            output = np.array(output_) * 255
            output = np.asarray(output, dtype='uint8')
            output2 = np.array(output2_) * 255
            output2 = np.asarray(output2, dtype='uint8')
            # output_img = Image.fromarray(output)
            # output_img.save(output_path + 'output_'+name)
            cv.imwrite(os.path.join(output_path[0], 'output1_' + name), output)
            cv.imwrite(os.path.join(output_path[1], 'output2_' + name), output2)
    except Exception as e:
        traceback.print_exc()
        print(e)


class Helper():
    def __init__(self, test_src_dir='/media/liu/File/Sp_320_dataset/tamper_result_320',
                 test_gt_dir='/media/liu/File/Sp_320_dataset/ground_truth_result_320'):
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
        gt_name = src_name.replace('Default', 'Gt').replace('png', 'bmp').replace('jpg', 'bmp')
        gt_path = os.path.join(self.test_gt_dir, gt_name)
        if os.path.exists(gt_path):
            pass
        else:
            print(gt_path, 'not exists')
            traceback.print_exc()
            sys.exit()
        return gt_path


if __name__ == '__main__':
    try:
        test_data_path = '/media/liu/File/Sp_320_dataset/tamper_result_320'
        output_path = ['/media/liu/File/10月数据准备/1108_数据测试/sp_train_data/1120stage_1_pred',
                       '/media/liu/File/10月数据准备/1108_数据测试/sp_train_data/1120stage_2_pred']
        if os.path.exists(output_path[0]):
            pass
        else:
            os.mkdir(output_path[0])

        if os.path.exists(output_path[1]):
            pass
        else:
            os.mkdir(output_path[1])
        model_path1 = '/home/liu/chenhaoran/Mymodel/record823/1111checkpoint8-stage1-0.296349-f10.863817-precision0.943144-acc0.995090-recall0.803569.pth'
        model_path2 = '/home/liu/chenhaoran/Mymodel/save_model119/1119checkpoint0-stage2-0.237375-f10.716162-precision0.938563-acc0.994791-recall0.597392.pth'
        checkpoint1 = torch.load(model_path1, map_location=torch.device('cpu'))
        checkpoint2 = torch.load(model_path2, map_location=torch.device('cpu'))
        model1 = Net1().to(device)
        model2 = Net2().to(device)
        # model = torch.load(model_path)
        model1.load_state_dict(checkpoint1['state_dict'])
        model2.load_state_dict(checkpoint2['state_dict'])
        model1.eval()
        model2.eval()
        read_test_data(output_path)
    except Exception as e:
        traceback.print_exc()
        print(e)
