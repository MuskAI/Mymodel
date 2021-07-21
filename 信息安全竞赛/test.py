"""
CREATED BY HAORAN
TIME: 2021-06-06

"""
# encoding=utf-8
import sys

sys.path.append('..')
import traceback
from two_stage_model import UNetStage1 as Net1
from two_stage_model import UNetStage2 as Net2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
import matplotlib
import random
import matplotlib.pyplot as plt
import cv2 as cv
import gc



class Test:
    def __init__(self, src_data_dir=None, output_dir=None):
        self.model_dir = output_dir
        self.output_dir = output_dir
        Test.read_test_data2(self, src_data_dir)

    def read_test_data2(self, src_data_dir):
        output_path = self.output_dir
        if os.path.exists(os.path.join(output_path, 'stage1')):
            print('existing: ', os.path.join(output_path, 'stage2'))
        else:
            os.mkdir(os.path.join(output_path, 'stage1'))
            os.mkdir(os.path.join(output_path, 'stage2'))
        output_path1 = os.path.join(output_path, 'stage1')
        output_path2 = os.path.join(output_path, 'stage2')
        try:
            image_name = os.listdir(src_data_dir)

            # TODO 2： 下面这行代码是对图片进行预处理，由于不同设备内存大小的影响
            #  当您的内存不够时，默认将普通width 和 height 减小一半
            for index, name in enumerate(tqdm(image_name)):
                image_path = os.path.join(src_data_dir, name)
                src = Image.open(image_path)
                if len(src.split()) == 4:
                    src = src.convert('RGB')
                img = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(src)
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
                        print('The error:', name)
                        print(model2_input.shape)
                        img = torchvision.transforms.Compose([
                            torchvision.transforms.Resize((src.size[0] // 2, src.size[1] // 2)),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                        ])(src)
                    print('resize:', (src.size[0] // 2, src.size[1] // 2))

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


                cv.imwrite(os.path.join(output_path1, (name.split('.')[0] + '.bmp')), output)
                cv.imwrite(os.path.join(output_path2, (name.split('.')[0] + '.bmp')), output2)
                del stage1_ouput, model2_input, output, output2
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            traceback.print_exc()
            print(e)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_path1 = './stage1.pth'
    model_path2 = './stage2.pth'
    checkpoint1 = torch.load(model_path1, map_location=device)
    checkpoint2 = torch.load(model_path2, map_location=device)
    model1 = Net1().to(device)
    model2 = Net2().to(device)

    model1.load_state_dict(checkpoint1['state_dict'])
    model2.load_state_dict(checkpoint2['state_dict'])
    model1.eval()
    model2.eval()

    # TODO 0 测试程序入口,测试结果将会保存在 './test_result' 中
    Test(src_data_dir='./test_data',output_dir='./test_result')

    # TODO 1 matplotlib画图

    # 判断是否完成
    if os.path.exists('./test_result/stage1') and os.path.exists('./test_result/stage2'):
        stage1_result_list = os.listdir(os.path.join('./test_result', 'stage1'))
        stage2_result_list = os.listdir(os.path.join('./test_result', 'stage2'))
        src_list = os.listdir('./test_data')
        stage1_result_list.sort(reverse=True)
        stage2_result_list.sort(reverse=True)
        src_list.sort(reverse=True)

        for idx in range(len(src_list)):
            src = Image.open(os.path.join('./test_data',src_list[idx]))
            s1 = Image.open(os.path.join('./test_result/stage1',stage1_result_list[idx]))
            s2 = Image.open(os.path.join('./test_result/stage2',stage2_result_list[idx]))
            if idx>=5:
                break

            plt.subplot(5,3,3*idx+1)
            plt.imshow(src)

            plt.subplot(5,3,3*idx+2)
            plt.imshow(s1)

            plt.subplot(5,3,3*idx+3)

            plt.imshow(s2)
        plt.show()
        plt.savefig('test_result.png')
    else:
        traceback.print_exc('测试未成功')