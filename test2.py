"""
created by haoran
time 8-17
"""
import traceback
from model import model
import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from functions import my_f1_score,my_accuracy_score,my_precision_score
import conf.global_setting as settings
#from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
from datasets.dataset import DataParser
from model.model import Net
from conf.global_setting import batch_size
import torchsummary as summary
import os, sys
import numpy as np
from PIL import Image
import shutil
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2 as cv
from functions import sigmoid_cross_entropy_loss, cross_entropy_loss,l1_loss,wce_huber_loss
from utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

def read_test_data():
    try:
        image_name = os.listdir(test_data_path)
        for name in image_name:
            print(image_name)
            name = 'Default_139_168593_refrigerator.png'
            gt_name = name.replace('Default','Gt')
            gt_name = gt_name.replace('png','bmp')
            gt_name = gt_name.replace('jpg','bmp')
            gt = Image.open(gt_name)
            gt = np.array(gt)
            plt.figure('gt')
            plt.imshow(gt)
            plt.show()

            image_path = os.path.join(test_data_path,name)
            img = Image.open('Default_139_168593_refrigerator.png')
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
        test_data_path = '/home/liu/chenhaoran/Mymodel/tes_820/tes_820'
        output_path = 'test_record/test_820/'
        model_path = './record/epoch-1-training-record.pth'
        model = torch.load(model_path)
        model = model.eval()
        print(model)
        model = model.cuda()
        read_test_data()
    except Exception as e:
        traceback.print_exc()
        print(e)
