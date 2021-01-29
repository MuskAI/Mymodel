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


        self.src_data_dir = self.COVERAGE_FOR_TRAIN_SRC
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
                output = model1(img)

                stage1_ouput = output[0]
                zero = torch.zeros_like(stage1_ouput)
                one = torch.ones_like(stage1_ouput)

                rgb_pred = img * torch.where(stage1_ouput > 0.1, one, zero)

                _rgb_pred = rgb_pred.squeeze(0)
                _rgb_pred = np.array(_rgb_pred)
                _rgb_pred = np.transpose(_rgb_pred, (1, 2, 0))

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

                output = np.array(output_) * 255
                output = np.asarray(output, dtype='uint8')
                output2 = np.array(output2_) * 255
                output2 = np.asarray(output2, dtype='uint8')

                cv.imwrite(os.path.join(output_path[0], 'output1_' + name), output)
                cv.imwrite(os.path.join(output_path[1], 'output2_' + name), output2)
        except Exception as e:
            traceback.print_exc()
            print(e)


if __name__ == '__main__':
    try:
        output_path = ['/media/liu/File/11月数据准备/1220test/casia_train_data_two_stage/1220stage_1_pred',
                       '/media/liu/File/11月数据准备/1220test/casia_train_data_two_stage/1220stage_2_pred']
        if os.path.exists(output_path[0]):
            pass
        else:
            os.mkdir(output_path[0])

        if os.path.exists(output_path[1]):
            pass
        else:
            os.mkdir(output_path[1])
        model_path1 = '/home/liu/chenhaoran/Mymodel/save_model/model_stage_one_casia_template_sp_train/1211_casia_template_sp_negative_checkpoint28-stage1-0.151637-f10.713804-precision0.859841-acc0.987976-recall0.627347.pth'
        # model_path1 = '/home/liu/chenhaoran/Mymodel/record823/1111checkpoint8-stage1-0.296349-f10.863817-precision0.943144-acc0.995090-recall0.803569.pth'
        model_path2 = '/home/liu/chenhaoran/Mymodel/save_model/1219_model_two_stage_band5_template_data/1119checkpoint4-stage2-0.090422-f10.569905-precision0.871056-acc0.992912-recall0.465043.pth'
        checkpoint1 = torch.load(model_path1,map_location=torch.device('cpu'))
        checkpoint2 = torch.load(model_path2, map_location=torch.device('cpu'))
        model1 = Net1().to(device)
        model2 = Net2().to(device)
        # model = torch.load(model_path)
        model1.load_state_dict(checkpoint1['state_dict'])
        model2.load_state_dict(checkpoint2['state_dict'])
        model1.eval()
        model2.eval()
        TestDataset()
    except Exception as e:
        traceback.print_exc()
        print(e)
