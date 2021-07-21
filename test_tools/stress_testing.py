"""
@author :haoran
time:0401
description:
这个文件是用来进行模糊压缩的压力测试的，输出是压力测试的结果图，输入是一个数据集的路径。
目的是实现在这个数据集上的压力测试
这里应该将测试时候的输入预处理和forward进行分开处理然后封装在一起
并将处理后的结果保存下来，期间还要对可能出现的图片读取错误进行异常处理
我的逻辑是对于每一个数据集就需要创建一个实例


我这里只计算第二阶段的指标，仅仅保存第一阶段的结果。
"""

import os,sys,traceback
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import torch,torchvision
import random
from tqdm import tqdm
import cv2 as cv
from model.unet_two_stage_model_0306 import UNetStage1 as Net1


from model.unet_two_stage_model_0306 import UNetStage2 as Net2

def get_jpeg(img, quality=90):
    img.save('temp_img_delete_it_if_you_want.jpg',quality=quality)
    img = Image.open('.temp_img_delete_it_if_you_want.jpg')
    return img

class AddJPEG(object):
    """
    jpeg
    """
    def __init__(self,quality):
        self.quality = quality

    def __call__(self, img):
        img.save('temp_img_delete_it_if_you_want.jpg', quality=self.quality)
        img = Image.open('temp_img_delete_it_if_you_want.jpg')
        return img
class AddGlobalBlur(object):
    """
    增加全局模糊t
    """

    def __init__(self, kernel_size=1.0):
        """
        :param p: p的概率会加模糊
        """
        self.kernel_size = kernel_size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """

        if self.kernel_size ==0:
            return img
        else:
            img_ = np.array(img).copy()
            img_ = Image.fromarray(img_)
            img_ = img_.filter(ImageFilter.GaussianBlur(radius=self.kernel_size))
            img_ = np.array(img_)
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
class StressTesting:
    def __init__(self,test_img_dir,output_dir,test_mode='blur'):
        self.test_img_dir = ''
        self.test_gt_dir = ''
        self.output_dir = ''
        self.blur_list = []
        self.compress_list = []
        self.test_mode = test_mode
        self.__setting()

        for idx,kernel_size in enumerate(tqdm(self.blur_list)):
            if idx <18:
                continue
            self.read_test_data2(src_data_dir=test_img_dir,kernel_size=kernel_size,output_path=output_dir)
        # for quality in tqdm(self.compress_list):
        #     self.read_test_data2(src_data_dir=test_img_dir,quality=quality,output_path=output_dir)

        pass
    def __setting(self):
        """
        首先是一些基本参数的设定
        :return:
        """
        self.blur_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
            ,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
        # self.blur_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
        self.compress_list = [100,95,90,85,80,75,70,65,60,55,50,45,40,35,30]
        self.save_percent = 1

        pass
    def start(self):
        pass
    def figure(self):
        """
        this is a test
        :return:
        """
        pass
    def save(self):
        pass
    # def read_test_data2(self,src_data_dir,kernel_size,output_path=None):
    #     test_data_path =src_data_dir
    #     output_stage1 = os.path.join(output_path, 'stage1_' + self.test_mode + '_'+str(kernel_size*10))
    #
    #     output_stage2 = os.path.join(output_path, 'stage2_' + self.test_mode + '_'+str(kernel_size*10))
    #     if os.path.exists(output_stage1):
    #         print('exists: ', output_stage1)
    #         pass
    #     else:
    #         os.mkdir(output_stage1)
    #         os.mkdir(output_stage2)
    #     output_path1 = output_stage1
    #     output_path2 = output_stage2
    #     try:
    #         image_name = os.listdir(test_data_path)
    #         for index, name in enumerate(tqdm(image_name)):
    #             image_path = os.path.join(test_data_path, name)
    #             img = Image.open(image_path)
    #             if len(img.split()) == 4:
    #                 img = img.convert('RGB')
    #
    #             if self.test_mode == 'normal':
    #                 img = torchvision.transforms.Compose([
    #                     torchvision.transforms.ToTensor(),
    #                     torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
    #                 ])(img)
    #             elif self.test_mode == 'blur':
    #                 img = torchvision.transforms.Compose([
    #                     AddGlobalBlur(kernel_size=kernel_size),
    #                     torchvision.transforms.ToTensor(),
    #                     torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
    #                 ])(img)
    #
    #             elif self.test_mode == 'compress':
    #
    #                 img = torchvision.transforms.Compose([
    #                     torchvision.transforms.ToTensor(),
    #                     torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
    #                 ])(img)
    #
    #             else:
    #                 traceback.print_exc('test_mode error!!!')
    #
    #
    #             img = img[np.newaxis, :, :, :]
    #             output = model1(img)
    #             stage1_ouput = output[0]
    #
    #             model2_input = torch.cat((stage1_ouput, img), 1)
    #
    #             output2 = model2(model2_input, output[1], output[2], output[3])
    #
    #             output = np.array(output[0].cpu().detach().numpy(), dtype='float32')
    #             output = output.squeeze(0)
    #             output = np.transpose(output, (1, 2, 0))
    #             output_ = output.squeeze(2)
    #
    #             output2 = np.array(output2[0].cpu().detach().numpy(), dtype='float32')
    #             output2 = output2.squeeze(0)
    #             output2 = np.transpose(output2, (1, 2, 0))
    #             output2_ = output2.squeeze(2)
    #
    #             output = np.array(output_) * 255
    #             output = np.asarray(output, dtype='uint8')
    #             output2 = np.array(output2_) * 255
    #             output2 = np.asarray(output2, dtype='uint8')
    #
    #             cv.imwrite(os.path.join(output_path1,name.split('.')[0]+'.bmp'), output)
    #             cv.imwrite(os.path.join(output_path2,name.split('.')[0]+'.bmp'), output2)
    #     except Exception as e:
    #         traceback.print_exc()
    #         print(e)
    def read_test_data2(self,src_data_dir,kernel_size=None,output_path=None,quality=90):
        test_data_path = src_data_dir
        if kernel_size!=None:
            output_stage1 = os.path.join(output_path, 'stage1_' + self.test_mode + '_' + str(int(kernel_size * 10)))

            output_stage2 = os.path.join(output_path, 'stage2_' + self.test_mode + '_' + str(int(kernel_size * 10)))
        else:
            output_stage1 = os.path.join(output_path, 'stage1_' + self.test_mode + '_' + str(quality))

            output_stage2 = os.path.join(output_path, 'stage2_' + self.test_mode + '_' + str(quality))
        if os.path.exists(output_stage1):
            print('exists: ', output_stage1)
            pass
        else:
            os.mkdir(output_stage1)
            os.mkdir(output_stage2)
        output_path1 = output_stage1
        output_path2 = output_stage2
        try:
            image_name = os.listdir(test_data_path)

            for index, name in enumerate(tqdm(image_name)):
                image_path = os.path.join(test_data_path, name)
                src = Image.open(image_path)
                if len(src.split()) == 4:
                    src = src.convert('RGB')

                if self.test_mode == 'normal':
                    img = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                    ])(src)
                elif self.test_mode == 'blur':
                    img = torchvision.transforms.Compose([
                        AddGlobalBlur(kernel_size=kernel_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                    ])(src)

                elif self.test_mode == 'compress':

                    img = torchvision.transforms.Compose([
                        AddJPEG(quality=quality),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                    ])(src)

                else:
                    traceback.print_exc('test_mode error!!!')

                for i in range(2):
                    try:
                        img = img[np.newaxis, :, :, :].cuda()
                        output = model1(img)
                        stage1_ouput = output[0].detach()

                        model2_input = torch.cat((stage1_ouput, img), 1).detach()

                        output2 = model2(model2_input, output[1], output[2], output[3])
                        output2[0].detach()
                        break
                    except Exception as e:
                        print('The error:',name)
                        print(model2_input.shape)
                        if self.test_mode == 'normal':
                            img = torchvision.transforms.Compose([
                                torchvision.transforms.Resize((src.size[0]//2,src.size[1]//2)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                            ])(src)
                        elif self.test_mode == 'blur':
                            img = torchvision.transforms.Compose([
                                AddGlobalBlur(kernel_size=kernel_size),

                                torchvision.transforms.Resize((src.size[0] // 2, src.size[1] // 2)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                            ])(src)

                        elif self.test_mode == 'compress':

                            img = torchvision.transforms.Compose([
                                AddJPEG(quality=quality),
                                torchvision.transforms.Resize((src.size[0] // 2, src.size[1] // 2)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                            ])(src)

                        else:
                            traceback.print_exc('test_mode error!!!')

                    print('resize:',(src.size[0]//2,src.size[1]//2))

                output = np.array(stage1_ouput.cpu().detach().numpy(), dtype='float32')

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
                # print(name.split('.')[0]+'.bmp')
                cv.imwrite(os.path.join(output_path1,  (name.split('.')[0]+'.bmp')), output)
                cv.imwrite(os.path.join(output_path2, (name.split('.')[0]+'.bmp')), output2)
                del stage1_ouput,model2_input,output,output2
                torch.cuda.empty_cache()
        except Exception as e:
            traceback.print_exc()
            print(e)

class CoverageTest(StressTesting):
    def __init__(self, src_data_dir=None, output_dir=None):
        self.src_data_dir = os.path.join(data_dir,'public_dataset/coverage')
        self.src_data_output_dir = os.path.join(output_dir, 'coverage_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        super(CoverageTest, self).__init__(test_img_dir=os.path.join(self.src_data_dir, 'src'),
                                     output_dir=self.src_data_train_output_dir)



class CasiaTest(StressTesting):
    def __init__(self, src_data_dir=None, output_dir=None):

        self.src_data_dir = os.path.join(data_dir,'public_dataset/casia')
        self.src_data_output_dir = os.path.join(output_dir, 'casia_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)


        super(CasiaTest, self).__init__(test_img_dir=os.path.join(self.src_data_dir, 'src'),
                                     output_dir=self.src_data_train_output_dir)


class ColumbiaTest(StressTesting):
    def __init__(self, src_data_dir=None, output_dir=None):

        self.src_data_dir = os.path.join(data_dir,'public_dataset/columbia')
        self.src_data_output_dir = os.path.join(output_dir, 'columbia_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)


        super(ColumbiaTest, self).__init__(test_img_dir=os.path.join(self.src_data_dir, 'src'),
                                     output_dir=self.src_data_train_output_dir)




if __name__ == '__main__':
    device = torch.device("cuda")
    try:
        output_path = '/home/liu/chenhaoran/test/0412_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束_stress_blur'
        if os.path.exists(output_path):
            pass
            # traceback.print_exc('The path is already exists ,please change it ')
        else:
            os.mkdir(output_path)
            print('mkdir :',output_path)
        model_path1 = '/home/liu/chenhaoran/Mymodel/save_model/0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束/stage1_0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束_checkpoint6-two_stage-0.047892-f10.890272-precision0.923245-acc0.993727-recall0.864488.pth'
        model_path2 = '/home/liu/chenhaoran/Mymodel/save_model/0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束/stage2_0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束_checkpoint6-two_stage-0.047892-f10.838588-precision0.884079-acc0.996823-recall0.802881.pth'
        checkpoint1 = torch.load(model_path1, map_location=torch.device('cuda'))
        checkpoint2 = torch.load(model_path2, map_location=torch.device('cuda'))
        model1 = Net1().to(device)
        model2 = Net2().to(device)
        # model = torch.load(model_path)
        model1.load_state_dict(checkpoint1['state_dict'])
        model2.load_state_dict(checkpoint2['state_dict'])
        model1.eval()
        model2.eval()
        data_device = '413'
        if data_device == '413':
            data_dir = '/home/liu/chenhaoran/Tamper_Data/3月最新数据'
        elif data_device == 'wkl':
            data_dir = r'D:\chenhaoran\data'

        try:
            # CasiaTest(output_dir=output_path)
            ColumbiaTest(output_dir=output_path)
            # CoverageTest(output_dir=output_path)
        except Exception as e:
            traceback.print_exc(e)
    except Exception as e:
        traceback.print_exc(e)