"""
created by haoran
time 1225
description:
希望通过这个文件实现比较智能的测试
输入只需要是网络结构和模型文件，输出是一份分析报告
1.
"""
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
device = torch.device("cpu")

class TestDataset:
    def __init__(self,src_data_dir=None,gt_data_dir=None,output_dir=None):
        self.src_data_dir = src_data_dir
        self.gt_data_dir = gt_data_dir
        self.model_dir = output_dir
        # sp dataset
        self.SP_DATA_FOR_TRAIN_SRC = '/media/liu/File/Sp_320_dataset/tamper_result_320'
        self.SP_DATA_FOR_TRAIN_GT = '/media/liu/File/Sp_320_dataset/ground_truth_result_320'

        # cm dataset
        self.CM_DATA_FOR_TRAIN_SRC = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
        self.CM_DATA_FOR_TRAIN_GT = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'

        # coverage dataset
        self.COVERAGE_FOR_TRAIN_SRC = '/media/liu/File/11月数据准备/AFTER_320_CROP_COVERAGE_DATASET/src'
        self.COVERAGE_FOR_TRAIN_GT = '/media/liu/File/11月数据准备/AFTER_320_CROP_COVERAGE_DATASET/gt'

        # casia template dataset
        self.TEMPLATE_FOR_TRAIN_SRC = '/media/liu/File/11月数据准备/CASIA_TEMPLATE_TRAIN/src'
        self.TEMPLATE_FOR_TRAIN_GT = '/media/liu/File/11月数据准备/CASIA_TEMPLATE_TRAIN/gt'

        # CASIA dataset
        self.TEST_CASIA2_DATA_FOR_TRAIN_SRC = '/media/liu/File/11月数据准备/CASIA2.0_DATA_FOR_TRAIN/src'
        self.SAVE_DIR = '/media/liu/File/11月数据准备/1211测试/casia_train_data/pred'

        self.NEGATIVE_FOR_TRAIN_SRC = '/media/liu/File/10月数据准备/10月12日实验数据/negative/src'
        self.NEGATIVE_FOR_TRAIN_GT = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'


        self.src_data_dir = self.TEST_CASIA2_DATA_FOR_TRAIN_SRC
        TestDataset.read_test_data(self)

    def read_test_data(self):
        test_data_path = self.src_data_dir

        try:
            image_name = os.listdir(test_data_path)
            length = len(image_name)
            for index, name in enumerate(image_name):
                print(index, '/', length)
                image_path = os.path.join(test_data_path, name)
                img = Image.open(image_path)
                # resize 的方式
                if img.size != (320, 320):
                    img = img.resize((320, 320))
                    img = np.array(img, dtype='uint8')

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
                output = model(img)
                output = output[0]

                output = np.array(output.cpu().detach().numpy(), dtype='float32')
                output = output.squeeze(0)

                output = np.transpose(output, (1, 2, 0))
                output_ = output.squeeze(2)

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
                cv.imwrite(os.path.join(output_path, 'output_' + name), output)
        except Exception as e:
            traceback.print_exc()
            print(e)
#
# class TestMultiDataset(TestDataset):
#     def __init__(self):
#         super(TestMultiDataset).__init__()


if __name__ == '__main__':
    model_path_0 = '/home/liu/chenhaoran/Mymodel/save_model/model_stage_one_casia_template_sp_train/1210checkpoint10-stage1-0.307417-f10.471496-precision0.486202-acc0.976723-recall0.488201.pth'
    model_path_1 = '/home/liu/chenhaoran/Mymodel/save_model/model_stage_one_casia_template_sp_train/1211_casia_template_sp_negative_checkpoint28-stage1-0.151637-f10.713804-precision0.859841-acc0.987976-recall0.627347.pth'
    model_path = model_path_1
    try:
        output_path = '/media/liu/File/11月数据准备/1220test/casia_train_data/pred'
        if os.path.exists(output_path):
            pass
        else:
            os.mkdir(output_path)

        checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
        model = Net().to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        TestDataset()
    except Exception as e:
        traceback.print_exc()
        print(e)