"""
created by HaoRan
time: 1114
description:
the only data reader
input: dataset path
output: a iterator
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from PIL import Image
import time
import os, sys
import traceback
from sklearn.model_selection import train_test_split
import cv2 as cv
import numpy as np
def gen_band_gt(gt):
    """
    01 mask 图省城边缘条带图
    :param gt: 01 mask图 输入的是list dim 4 维，B C H W
    :return:list 01 边缘条带,list B C H W
    """
    band_gt = gt
    for i in range(gt.shape[0]):
        _gt = gt[i,:,:,:]
        _gt = _gt.squeeze(0)
        _gt = cv.cvtColor(np.asarray(_gt), cv.COLOR_GRAY2BGR)
        _gt = np.array(_gt,dtype='uint8')
        cv2_gt = cv.cvtColor(_gt, cv.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        cv2_gt = cv.dilate(cv2_gt, kernel)
        _band = Image.fromarray(cv.cvtColor(cv2_gt, cv.COLOR_BGR2RGB))
        _band = np.array(_band)[:,:,0]
        band_gt[i,:,:,:] = np.expand_dims(_band,0)

    return band_gt

class TamperDataset(Dataset):
    def __init__(self, stage_type='stage1', transform=None, train_val_test_mode='train', device='413', using_data=None, val_percent=0.1):
        """
        The only data loader for train val test dataset
        using_data = {'my_sp':True,'my_cm':True,'casia':True,'copy_move':True,'columb':True}
        :param transform: only for src transform
        :param train_val_test_mode: the type is string
        :param device: using this to debug, 413, laptop for choose
        :param using_data: a dict, e.g.
        """

        # train val test mode
        self.train_val_test_mode = train_val_test_mode
        self.stage_type = stage_type

        # if the mode is train ,split it to get val
        """train or test mode"""
        if train_val_test_mode == 'train' or 'val':


            train_val_src_list, train_val_gt_list, train_val_band_gt_list = MixData(train_mode=True,using_data=using_data).gen_dataset()

            self.train_src_list, self.val_src_list, self.train_gt_list, self.val_gt_list = \
                train_test_split(train_val_src_list, train_val_gt_list, test_size=val_percent, train_size=1-val_percent, random_state=1000)

            _train_src_list, _val_src_list, self.train_band_gt_list, self.val_band_gt_list = \
                train_test_split(train_val_src_list, train_val_band_gt_list, test_size=val_percent, train_size=1-val_percent,random_state=1000)


            self.transform = transform
            # if there is a check function would be better
        elif train_val_test_mode == 'test':
            self.test_src_list, self.test_gt_list, self.test_band_gt_list = MixData().gen_dataset()
        else:
            raise EOFError

    def __getitem__(self, index):
        """
        train val test 区别对待
        :param index:
        :return:
        """
        # train mode
        # val mode
        # test mode
        mode = self.train_val_test_mode

        # default mode
        tamper_path = self.train_src_list[index]
        gt_path = self.train_gt_list[index]
        gt_band_path = self.train_band_gt_list[index]
        if mode == 'train':

            tamper_path = self.train_src_list[index]
            gt_path = self.train_gt_list[index]
            gt_band_path = self.train_band_gt_list[index]

        elif mode == 'val':

            tamper_path = self.val_src_list[index]
            gt_path = self.val_gt_list[index]
            gt_band_path = self.val_band_gt_list[index]

        elif mode == 'test':

            tamper_path = self.test_src_list[index]
            gt_path = self.test_gt_list[index]
            gt_band_path = self.test_band_gt_list[index]
        else:
            traceback.print_exc('an error occur')
        # read img
        img = Image.open(tamper_path)
        gt = Image.open(gt_path)
        try:
            gt_band = Image.open(gt_band_path)
        except Exception as e:
            gt_band = TamperDataset.__gen_band(self, gt)
        # transform
        gt = transforms.ToTensor()(gt)
        gt_band = transforms.ToTensor()(gt_band)

        # if transform src
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)


        if self.stage_type == 'stage1':
            sample = {'tamper_image': img, 'gt_band': gt_band}
        else:
            sample = {'tamper_image': img, 'gt_band': gt_band}
        return sample

    def __len__(self):
        mode = self.train_val_test_mode
        length = len(self.train_src_list)
        if mode == 'train':
            length = len(self.train_src_list)
        elif mode == 'val':
            length = len(self.val_src_list)

        elif mode == 'test':
            length = len(self.test_src_list)
        else:
            traceback.print_exc('an error occur')
        return length

    def __gen_band(self,gt, dilate_window = 5):
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

        _gt = np.array(_gt,dtype='uint8')

        if max(_gt.reshape(-1)) == 255:
            _gt = np.where((_gt == 255) | (_gt == 100), 1, 0)
            _gt = np.array(_gt, dtype='uint8')
        else:
            pass

        _gt = cv.merge([_gt])
        kernel = np.ones((dilate_window, dilate_window), np.uint8)
        _band = cv.dilate(_gt, kernel)
        _band = Image.fromarray(np.array(_band,dtype='uint8'))
        if len(_band.split()) == 3:
            _band = np.array(_band)[:, :, 0]
        else:
            _band = np.array(_band)
        return _band


class MixData:
    def __init__(self, train_mode=True, using_data=None,device='413'):

        gt_path_list = []
        data_dict = MixData.__data_path_gather(self,train_mode=train_mode, using_data=using_data,device=device)
        self.src_path_list = data_dict['src_path_list']
        self.sp_gt_path = data_dict['sp_gt_path']
        self.sp_gt_band_path = data_dict['sp_gt_band_path']
        self.cm_gt_path = data_dict['cm_gt_path']
        self.cm_gt_band_path = data_dict['cm_gt_band_path']
        self.negative_gt_path = data_dict['negative_gt_path']
        # if True:
        #     self.src_path_list = ['/media/liu/File/debug_data/tamper_result']
        #     self.cm_gt_path = '/media/liu/File/debug_data/ground_truth_r esult'

    def gen_dataset(self):
        """
        通过输入的src & gt的路径生成train_list 列表
        并通过check方法，检查是否有误
        :return:
        """
        dataset_type_num = len(self.src_path_list)
        train_list = []
        gt_list = []
        gt_band_list = []
        unmatched_list = []
        # 首先开始遍历不同类型的数据集路径
        for index1, item1 in enumerate(self.src_path_list):
            for index2,item2 in enumerate(os.listdir(item1)):
                t_img_path = os.path.join(item1, item2)
                t_gt_path, t_gt_band_path = MixData.__switch_case(self, t_img_path)
                if t_gt_path != '' or t_gt_band_path != '':
                    train_list.append(t_img_path)
                    gt_list.append(t_gt_path)
                    gt_band_list.append(t_gt_band_path)

                else:
                    print(t_gt_path, t_gt_path,'unmatched')
                    unmatched_list.append([t_img_path,t_gt_path])
                    print('The process: %d/%d : %d/%d'%(index1+1, len(self.src_path_list), index2+1,len((os.listdir(item1)))))
        print('The number of unmatched data is :', len(unmatched_list))
        print('The unmatched list is : ',unmatched_list)


        # if MixData.__check(self,train_list=, gt_list=):
        #     pass
        # else:
        #     print('check error, please redesign')
        #     traceback.print_exc()
        #     sys.exit()

        return train_list, gt_list, gt_band_list

    def __check(self, train_list, gt_list):
        """
        检查train_list 和 gt_list 是否有问题
        :return:
        """
        pass

    def __switch_case(self, path, band_type='_band5'):
        """
        针对不同类型的数据集做处理
        :return: 返回一个路径，这个路径是path 所对应的gt路径，并且需要检查该路径是否存在
        """
        # 0 判断路径的合法性
        if os.path.exists(path):
            pass
        else:
            print('The path :', path, 'does not exist')
            return ''
        # 1 分析属于何种类型
        # there are
        # 1.  sp generate data
        # 2. cm generate data
        # 3. negative data
        # 4. CASIA data

        sp_type = ['Sp']
        cm_type = ['Default','poisson']
        negative_type = ['negative']
        CASIA_type = ['Tp']
        debug_type = ['debug']
        type= []
        name = path.split('/')[-1]
        # name = path.split('\\')[-1]
        for sp_flag in sp_type:
            if sp_flag in name[:2]:
               type.append('sp')
               break

        for cm_flag in cm_type:
            if cm_flag in name[:7]:
                type.append('cm')
                break

        for negative_flag in negative_type:
            if negative_flag in name:
                type.append('negative')
                break

        for CASIA_flag in CASIA_type:
            if CASIA_flag in name[:2]:
                type.append('casia')
                break


        # 判断正确性

        # gt_band 统一在input_img_name_后面加一个_band5.bmp
        if len(type) != 1:
            print('The type len is ', len(type))
            return ''

        if type[0] == 'sp':
            gt_path = name.replace('Default','Gt').replace('.jpg','.bmp').replace('.png', '.bmp').replace('poisson','Gt')
            gt_band_path = gt_path.split('.')[0]+band_type+'.bmp'
            gt_path = os.path.join(self.sp_gt_path, gt_path)
            gt_band_path = os.path.join(self.sp_gt_band_path, gt_band_path)
        elif type[0] == 'cm':
            gt_path = name.replace('Default', 'Gt').replace('.jpg','.bmp').replace('.png', '.bmp').replace('poisson','Gt')
            gt_band_path = gt_path.split('.')[0] + band_type + '.bmp'
            gt_path = os.path.join(self.cm_gt_path, gt_path)
            gt_band_path = os.path.join(self.cm_gt_band_path, gt_band_path)

        elif type[0] == 'negative':
            gt_path = 'negative_gt.bmp'
            gt_band_path = gt_path.split('.')[0] + band_type + '.bmp'
            gt_path = os.path.join(self.negative_gt_path, gt_path)
            gt_band_path = os.path.join(self.negative_gt_band_path, gt_band_path)
        elif type[0] == 'casia':
            gt_path = name.split('.')[0] + '_gt' + '.png'
            gt_band_path = gt_path.split('.')[0] + band_type + '.bmp'
            gt_path = os.path.join(self.casia_gt_path, gt_path)
            gt_band_path = os.path.join(self.casia_gt_band_path, gt_band_path)
        else:
            print('Error')
            sys.exit()
        # 判断gt是否存在
        if os.path.exists(gt_path):
            pass
        else:
            return ''

        return gt_path, gt_band_path

    def __data_path_gather(self, train_mode=True, device='413', using_data=None):
        """
        using_data = {'my_sp':True,'my_cm':True,'casia':True,'copy_move':True,'columb':True,'negative':True}
        :param device:
        :param using_data:
        :return:
        """
        src_path_list = []
        if using_data:
            pass
        else:
            traceback.print_exc('using_data input None error')
        self.sp_gt_path = None
        self.sp_gt_band_path = None
        self.cm_gt_path = None
        self.cm_gt_band_path = None
        self.negative_gt_path = None
        self.casia_gt_band_path = None
        self.casia_gt_path = None


        if device == '413':
            if using_data['my_sp']:
                if train_mode:
                    path = '/media/liu/File/Sp_320_dataset/tamper_result_320'
                    src_path_list.append(path)
                    self.sp_gt_path = '/media/liu/File/Sp_320_dataset/ground_truth_result_320'
                    self.sp_gt_band_path = ''

                else:
                    path = '/media/liu/File/Sp_320_dataset/tamper_result_320'
                    src_path_list.append(path)
                    self.sp_gt_path = '/media/liu/File/Sp_320_dataset/ground_truth_result_320'
                    self.sp_gt_band_path = ''
            if using_data['my_cm']:
                if train_mode:
                    path = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
                    src_path_list.append(path)
                    self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'
                    self.cm_gt_band_path = ''
                else:
                    path = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
                    src_path_list.append(path)
                    self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'
                    self.cm_gt_band_path = ''
            if using_data['casia']:
                if train_mode:
                    path = '/media/liu/File/Sp_320_dataset/tamper_result_320'
                    src_path_list.append(path)
                    self.casia_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/casia/gt'
                    self.casia_gt_band_path=''
                else:
                    path = '/media/liu/File/Sp_320_dataset/tamper_result_320'
                    src_path_list.append(path)
                    self.casia_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/casia/gt'
                    self.casia_gt_band_path=''
            if using_data['copy_move']:
                path = '/media/liu/File/Sp_320_dataset/tamper_result_320'
                src_path_list.append(path)
            if using_data['columb']:
                path = '/media/liu/File/Sp_320_dataset/tamper_result_320'
                src_path_list.append(path)
            if using_data['negative']:
                if train_mode:
                    path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/src'
                    src_path_list.append(path)
                    self.negative_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'
                else:
                    path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/src'
                    src_path_list.append(path)
                    self.negative_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'
        elif device == 'flyai':
            pass


        self.src_path_list = src_path_list
        return {'src_path_list':self.src_path_list,
                'sp_gt_path':self.sp_gt_path,
                'sp_gt_band_path':self.sp_gt_band_path,
                'cm_gt_path': self.cm_gt_path,
                'cm_gt_band_path':self.cm_gt_band_path,
                'casia_gt_path':self.casia_gt_path,
                'casia_gt_band_path': self.casia_gt_band_path,
                'negative_gt_path': self.negative_gt_path
                }

if __name__ == '__main__':
    print('start')
    mydataset = TamperDataset(using_data={'my_sp':True,'my_cm':False,'casia':False,'copy_move':False,'columb':False, 'negative': False})
    dataloader = torch.utils.data.DataLoader(mydataset,batch_size=10,num_workers=2)
    start = time.time()
    for idx, item in enumerate(dataloader):
        print(idx,type(item))
        print(len(dataloader))
        if idx == 3000 :
            break
    end = time.time()
    print('time :', end-start)