"""
created by haoran
time:8-24
使用其他尺寸的数据集
"""
import os
import cv2 as cv
from PIL import Image
import sys
import traceback
import numpy as np


def using_coverage_data(coverage_data_path):
    image_path = os.path.join(coverage_data_path,'image')
    mask_path = os.path.join(coverage_data_path,'mask')



class CoverageData():
    def __init__(self,coverage_data_path,target_size = (320,320)):
        self.coverage_data_path = coverage_data_path
        self.image_path = os.path.join(coverage_data_path,'image')
        self.mask_path = os.path.join(coverage_data_path,'mask')
        self.save_path = 'C:\\Users\\musk\\Desktop\\coverage\\save_forged'
        self.save_path_img = ''
        self.save_path_mask = ''
        self.target_size = target_size
    def size_ok(self,real_size):
        if real_size[0] > self.target_size[0] and real_size[1] > self.target_size[1]:
            return True
        else:
            return False
    def random_crop(self,img,mask,real_size):
        permit_row_range = real_size[0] - self.target_size[0]
        permit_col_range = real_size[1] - self.target_size[1]
        random_y = np.random.randint(0,permit_row_range)
        random_x = np.random.randint(0,permit_col_range)
        box = (random_y,random_x,random_y+self.target_size[0],random_x + self.target_size[1])
        crop_img = img.crop(box)
        crop_mask = mask.crop(box)
        return crop_img, crop_mask

    def find_tamper_data(self):
        """

        :return: tamper image list , tamper mask list
        """
        tamper_img_list = []
        mask_list = []
        for t_img in os.listdir(self.image_path):
            if 't' in t_img.replace('.tif',''):
                tamper_img_list.append(os.path.join(self.image_path, t_img))
                mask_list.append(os.path.join(self.mask_path,t_img.replace('.tif','').replace('t','forged')+'.tif'))
            else:
                continue

        return tamper_img_list, mask_list
    def gen_tamper_data(self):
        self.save_path_img = os.path.join(self.save_path, 'forged_img')
        self.save_path_mask = os.path.join(self.save_path, 'forged_mask')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

            os.mkdir(self.save_path_img)
            os.mkdir(self.save_path_mask)
        else:
            pass

        tamper_img, tamper_mask = self.find_tamper_data()

        for index, img in enumerate(tamper_img):

            print(img)
            print(tamper_mask[index])

            t_img = cv.imread(img,1)
            cv.namedWindow('img')
            cv.imshow('img', t_img)
            cv.waitKey(100)
            t_img = Image.fromarray(cv.cvtColor(t_img,cv.COLOR_BGR2RGB))

            t_mask = cv.imread(tamper_mask[index], 1)
            cv.namedWindow('mask')
            cv.imshow('mask', t_mask)
            cv.waitKey(100)
            t_mask = Image.fromarray(cv.cvtColor(t_mask, cv.COLOR_BGR2RGB))
            if not self.size_ok(t_img.size):
                print('该图片大小不符合，跳过，下一张')
                continue
            else:
                t_img_random_crop, t_mask_random_crop = self.random_crop(t_img, t_mask,real_size=t_img.size)
                t_img_random_crop.save(os.path.join(self.save_path_img,'%d.png'%index))
                t_mask_random_crop.save(os.path.join(self.save_path_mask,'%d.png'%index))
                print('%d/%d'%(index, len(tamper_img)))

if __name__ == '__main__':
    CoverageData('C:\\Users\\musk\\Desktop\\coverage').gen_tamper_data()