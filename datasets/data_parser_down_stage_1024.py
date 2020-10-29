import os
import numpy as np
from PIL import Image, ImageFilter
from image_squeene import compress_image, get_size, MyGaussianBlur
import random
from sklearn.model_selection import train_test_split
from PIL import ImageFile
import traceback
import cv2
import sys
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
class DataParser():
    def __init__(self, batch_size_train):
        self.ground_list = []
        self.trainimage_list = []
        self.batch_size = batch_size_train
        self.gt_list = []
        self.train_image_list = []
        self.dou_edge_list = []
        self.final_dou_edge_list = []
        self.X_train_or_test = []
        self.Y_train_or_test = []


        self.train_image_list, self.gt_list = MixData().gen_dataset()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.train_image_list, self.gt_list, test_size=0.1, train_size=0.9,random_state=1300)


        self.steps_per_epoch = len(self.X_train) / batch_size_train
        self.val_steps = len(self.X_test) / (batch_size_train)

        self.image_width = 320
        self.image_height = 320

        self.target_regression = True

    def get_batch(self, batch, train=True):

        filenames = []
        images = []
        edgemaps = []
        double_edge = []
        edgemaps_4 = []
        edgemaps_8 = []
        edgemaps_16 = []
        chanelfuse = []
        chanel = [[] for i in range(8)]
        for idx, b in enumerate(batch):
            if train:
                self.X_train_or_test = self.X_train
                self.Y_train_or_test = self.Y_train
            else:
                self.X_train_or_test = self.X_test
                self.Y_train_or_test = self.Y_test

            # train data
            index = self.X_train_or_test.index(b)
            im = Image.open(self.X_train_or_test[index])

            # train gt
            dou_path = self.Y_train_or_test[index]
            dou_em = Image.open(dou_path)
            if len(dou_em.split()) == 3:
                dou_em = dou_em.split()[0]
                # print('dim error')
            else:
                pass
            # 在这里获取8张图，从左上角按照顺时针顺序,返回的是一个长度为8的列表, 类型默认为Image
            relation_8_map = DataParser.gen_8_2_map(self,np.array(dou_em))
            # im, dou_em = DataParser.combine_augment_method(self,im,dou_em,relation_8_map,index = index)
            im = np.array(im, dtype=np.float32)
            im = im[..., ::-1]  # RGB 2 BGR
            # R=118.98194217348079 G=127.4061956623793 B=138.00865419127499
            im[..., 0] -= 138.008
            im[..., 1] -= 127.406
            im[..., 2] -= 118.982

            dou_chanel = [0 for i in range(8)]
            for i in range(8):
                dou_chanel[i] = np.array(relation_8_map[i], dtype=np.float32)
                dou_chanel[i] = np.array(dou_chanel[i][:, :, 1:])
                # dou_chanel[i] = dou_chanel[i] / 255
                chanel[i].append(dou_chanel[i])

            c_1 = dou_chanel[0][:, :, 1:]
            c_2 = dou_chanel[1][:, :, 1:]
            c_3 = dou_chanel[2][:, :, 1:]
            c_4 = dou_chanel[3][:, :, 1:]
            c_5 = dou_chanel[4][:, :, 1:]
            c_6 = dou_chanel[5][:, :, 1:]
            c_7 = dou_chanel[6][:, :, 1:]
            c_8 = dou_chanel[7][:, :, 1:]
            final_c = np.concatenate((c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8), axis=2)
            chanelfuse.append(final_c)

            if os.path.exists(os.path.join('/media/liu/File/10月数据准备/10月12日实验数据/cm/band_save',
                                                 self.Y_train_or_test[index].split('/')[-1].split('.')[
                                                     0] + '_band_gt.' +
                                                 self.Y_train_or_test[index].split('/')[-1].split('.')[1])):

                dou_em = Image.open(os.path.join('/media/liu/File/10月数据准备/10月12日实验数据/cm/band_save',
                                                 self.Y_train_or_test[index].split('/')[-1].split('.')[
                                                     0] + '_band_gt.' +
                                                 self.Y_train_or_test[index].split('/')[-1].split('.')[1]))

                if len(dou_em.split()) ==3:
                    dou_em = dou_em.split()[0]
                else:
                    pass
            else:
                pass

            dou_em = np.array(dou_em, dtype=np.float32)

            # 转化为无类别的GT 100 255 为边缘
            dou_em = np.where(dou_em == 50,0,dou_em)
            dou_em = np.where(dou_em ==100,1,dou_em)
            dou_em = np.where(dou_em == 255, 1, dou_em)



            dou_em = np.expand_dims(dou_em, 2)

            double_edge.append(dou_em)

            t_4 = dou_em.squeeze(2)
            t_4 = Image.fromarray(t_4)
            t_4 = t_4.resize((40,40),Image.BICUBIC)
            t_4 = np.where(np.array(t_4)>0,1,0)
            t_4 = np.expand_dims(t_4,2)
            edgemaps_4.append(t_4)
            t_8 = dou_em.squeeze(2)
            t_8 = Image.fromarray(t_8)
            t_8 = t_8.resize((80, 80), Image.BICUBIC)
            t_8 = np.where(np.array(t_8) > 0, 1, 0)
            t_8 = np.expand_dims(t_8, 2)
            edgemaps_8.append(t_8)
            t_16 = dou_em.squeeze(2)
            t_16 = Image.fromarray(t_16)
            t_16 = t_16.resize((160, 160), Image.BICUBIC)
            t_16 = np.where(np.array(t_16) > 0, 1, 0)
            t_16 = np.expand_dims(t_16, 2)
            edgemaps_16.append(t_16)
            images.append(im)
            filenames.append(self.X_train_or_test[index])

        images = np.asarray(images)
        double_edge = np.asarray(double_edge)
        edgemaps_4 = np.asarray(edgemaps_4)
        edgemaps_8 = np.asarray(edgemaps_8)
        edgemaps_16 = np.asarray(edgemaps_16)

        chanel1 = np.asarray(chanel[0])
        chanel2 = np.asarray(chanel[1])
        chanel3 = np.asarray(chanel[2])
        chanel4 = np.asarray(chanel[3])
        chanel5 = np.asarray(chanel[4])
        chanel6 = np.asarray(chanel[5])
        chanel7 = np.asarray(chanel[6])
        chanel8 = np.asarray(chanel[7])
        chanelfuse = np.asarray(chanelfuse)

        return images, edgemaps, double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, chanelfuse, edgemaps_4, edgemaps_8, edgemaps_16, filenames

    def multi_scale(self):
        pass

    def combine_augment_method(self, im, dou_em, relation_8_map, index):
        """
        基于彪哥给的代码做修改
        :param im:
        :return:
        """
        weight = im.size[0]
        height = im.size[1]

        if im.size[0] > 320 or im.size[1] > 320:
            print('The input image path size is error: ', im.size[0], im.size[1])
            sys.exit('please check the code')

        else:
            # 1 决定图像是否加压缩
            if random.randint(0, 20) == 1:
                try:
                    mb = random.randint(30, 100)
                    path = compress_image(infile=self.X_train_or_test[index], mb=mb)
                    im = Image.open(path)
                except:
                    traceback.print_exc()

            # 2 决定图像是否加模糊
            if random.randint(0, 20) == 1:
                m = random.randint(0, 5)
                if m == 0:
                    r = random.randint(1, 3)
                    im = im.filter(ImageFilter.GaussianBlur(radius=r))
                else:
                    pass
            # 3 决定是否flip 旋转
            random_aug = random.randint(0, 20)
            try:
                if random_aug < 5:
                    if random_aug == 0:
                        # 旋转90
                        im = im.transpose(Image.ROTATE_90)
                        dou_em = dou_em.transpose(Image.ROTATE_90)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.ROTATE_90)

                    elif random_aug == 1:
                        # 旋转180
                        im = im.transpose(Image.ROTATE_180)
                        dou_em = dou_em.transpose(Image.ROTATE_180)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.ROTATE_180)

                    elif random_aug == 2:
                        # 旋转270
                        im = im.transpose(Image.ROTATE_270)
                        dou_em = dou_em.transpose(Image.ROTATE_270)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.ROTATE_270)
                    elif random_aug == 3:
                        # 左右互换
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)
                        dou_em = dou_em.transpose(Image.FLIP_LEFT_RIGHT)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.FLIP_LEFT_RIGHT)
                    elif random_aug == 4:
                        # 左右呼唤
                        im = im.transpose(Image.FLIP_TOP_BOTTOM)
                        dou_em = dou_em.transpose(Image.FLIP_TOP_BOTTOM)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.FLIP_TOP_BOTTOM)
            except:
                traceback.print_exc()
        return im, dou_em

    def __src_check(self):
        """
        检查src是否有效
        :return:
        """
        pass

    def __gt_check(self):
        """
        检查GT是否有效
        :return:
        """
        pass

    def debug_test(self):
        """
        开发的时候用来测
        :return:
        """
        src_path = r'C:\Users\musk\Desktop\tamper_test\Default_78_398895_clock.png'
        gt_path = r'C:\Users\musk\Desktop\tamper_test\Gt_78_398895_clock.bmp'
        self.X_train_or_test = [str(src_path)]
        src = Image.open(src_path)
        gt = Image.open(gt_path)
        relation_8_map = DataAugment.gen_8_2_map(self, np.array(gt))
        for i in range(100):
            im, gt = DataAugment.combine_augment_method(self, src, gt, relation_8_map=relation_8_map, index=0)
            im.save(r'C:\Users\musk\Desktop\tamper_test\augment_test' + '\\' + str(i) + '_src.png')
            gt.save(r'C:\Users\musk\Desktop\tamper_test\augment_test' + '\\' + str(i) + '_gt.bmp')

    def gen_8_2_map(self, mask, mask_area=50, mask_edge=255, not_mask_edge=100, not_mask_area=0, output_type='Image'):
        """
        输入mask，先按照固定参数标好，篡改区域、篡改区域边缘，非篡改区域边缘，非篡改区域
        :param mask: 255 的图，channel数为1
        :return: 从左上角的点开始按照顺时针方向的8张二通道的图
        """
        # 在输入mask 之前对mask进行检查
        if type(mask) is np.ndarray:
            if mask.ndim == 2:
                pass
            else:
                # 如果输入的维度不是3二维的，则转化成2 dim
                print('Notice: when using the function gen_8_2_map, the mask ndim is not 1 but', mask.ndim)
                mask = mask[:, :, 0]
        else:
            print('Notice: when using function gen_8_2_map, the input mask not numpy array')
            traceback.print_exc()
            sys.exit()

        # 开始进行8张图的生成
        relation_8_map = []
        edge_loc_ = [1, 1]
        # 找到内侧和外侧边缘
        mask_pad = np.pad(mask, (1, 1), mode='constant')
        mask_pad = np.where(mask_pad == 50, 0, mask_pad)
        edge_loc = np.where(mask_pad == mask_edge)
        edge_loc_1 = np.where(mask_pad == not_mask_edge)
        edge_loc_[0] = np.append(edge_loc[0], edge_loc_1[0])
        edge_loc_[1] = np.append(edge_loc[1], edge_loc_1[1])

        del edge_loc_1
        del edge_loc
        edge_loc = edge_loc_
        mask_shape = mask_pad.shape
        # 生成8张结果图
        for i in range(8):
            temp = np.ones((mask_shape[0], mask_shape[1], 2))
            relation_8_map.append(temp)

        for j in range(len(edge_loc[0])):
            row = edge_loc[0][j]
            col = edge_loc[1][j]
            if mask_pad[row - 1, col - 1] != 0:
                relation_8_map[0][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row - 1, col - 1]:
                    relation_8_map[0][row, col, 1] = 1
                else:
                    relation_8_map[0][row, col, 1] = 0
            else:
                relation_8_map[0][row, col, 0] = 0
                relation_8_map[0][row, col, 1] = 0

            if mask_pad[row - 1, col] != 0:
                relation_8_map[1][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row - 1, col]:
                    relation_8_map[1][row, col, 1] = 1
                else:
                    relation_8_map[1][row, col, 1] = 0
            else:
                relation_8_map[1][row, col, 0] = 0
                relation_8_map[1][row, col, 1] = 0

            if mask_pad[row - 1, col + 1] != 0:
                relation_8_map[2][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row - 1, col + 1]:
                    relation_8_map[2][row, col, 1] = 1
                else:
                    relation_8_map[2][row, col, 1] = 0
            else:
                relation_8_map[2][row, col, 0] = 0
                relation_8_map[2][row, col, 1] = 0

            if mask_pad[row, col + 1] != 0:
                relation_8_map[3][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row, col + 1]:
                    relation_8_map[3][row, col, 1] = 1
                else:
                    relation_8_map[3][row, col, 1] = 0
            else:
                relation_8_map[3][row, col, 0] = 0
                relation_8_map[3][row, col, 1] = 0

            if mask_pad[row + 1, col + 1] != 0:
                relation_8_map[4][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row + 1, col + 1]:
                    relation_8_map[4][row, col, 1] = 1
                else:
                    relation_8_map[4][row, col, 1] = 0
            else:
                relation_8_map[4][row, col, 0] = 0
                relation_8_map[4][row, col, 1] = 0

            if mask_pad[row + 1, col] != 0:
                relation_8_map[5][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row + 1, col]:
                    relation_8_map[5][row, col, 1] = 1
                else:
                    relation_8_map[5][row, col, 1] = 0
            else:
                relation_8_map[5][row, col, 0] = 0
                relation_8_map[5][row, col, 1] = 0

            if mask_pad[row + 1, col - 1] != 0:
                relation_8_map[6][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row + 1, col - 1]:
                    relation_8_map[6][row, col, 1] = 1
                else:
                    relation_8_map[6][row, col, 1] = 0
            else:
                relation_8_map[6][row, col, 0] = 0
                relation_8_map[6][row, col, 1] = 0

            if mask_pad[row, col - 1] != 0:
                relation_8_map[7][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row, col - 1]:
                    relation_8_map[7][row, col, 1] = 1
                else:
                    relation_8_map[7][row, col, 1] = 0
            else:
                relation_8_map[7][row, col, 0] = 0
                relation_8_map[7][row, col, 1] = 0

        for i in range(8):
            relation_8_map[i] = relation_8_map[i][1:-1, 1:-1, :]
            # plt.figure('123')
            # plt.imshow(relation_8_map[i][:,:,0])
            # plt.savefig('C:\\Users\\musk\\Desktop\\边缘8张图关系图\\%d.png'%i)
            # plt.show()
            # # temp = Image.fromarray(relation_8_map[i])
            # # temp = temp.convert('RGB')
            # # temp.save('C:\\Users\\musk\\Desktop\\边缘8张图关系图\\%d.png'%i)
        if output_type == 'Image':
            for i in range(8):
                relation_8_map[i] = Image.fromarray(relation_8_map[i].astype('uint8')).convert('RGB')
        else:
            pass

        return relation_8_map

def generate_minibatches(dataParser, train=True):

    while True:
        if train:
            batch_ids = np.random.choice(dataParser.X_train, dataParser.batch_size)
            ims, ems, double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, chanel_fuse, edgemaps_4, edgemaps_8, edgemaps_16, _ = dataParser.get_batch(
                batch_ids)
        else:
            batch_ids = np.random.choice(dataParser.X_test, dataParser.batch_size)
            ims, ems, double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, chanel_fuse, edgemaps_4, edgemaps_8, edgemaps_16, _ = dataParser.get_batch(
                batch_ids, train=False)

        # datagen.flow()
        yield (ims, [chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, ems, ems])


class DataAugment():
    def __init__(self):
        pass

    def combine_augment_method(self, im, dou_em, relation_8_map, index):
        """
        基于彪哥给的代码做修改
        :param im:
        :return:
        """
        weight = im.size[0]
        height = im.size[1]

        if im.size[0] > 320 or im.size[1] > 320:
            print('The input image path size is error: ',im.size[0],im.size[1])
            sys.exit('please check the code')

        else:
            # 1 决定图像是否加压缩
            if random.randint(0, 20) == 1:
                try:
                    mb = random.randint(30, 100)
                    path = compress_image(infile=self.X_train_or_test[index], mb=mb)
                    im = Image.open(path)
                except:
                    traceback.print_exc()

            # 2 决定图像是否加模糊
            if random.randint(0, 20) == 1:
                m = random.randint(0, 5)
                if m == 0:
                    r = random.randint(1, 3)
                    im = im.filter(ImageFilter.GaussianBlur(radius=r))
                else:
                    pass
            # 3 决定是否flip 旋转
            random_aug = random.randint(0, 20)
            try:
                if random_aug < 5:
                    if random_aug == 0:
                        # 旋转90
                        im = im.transpose(Image.ROTATE_90)
                        dou_em = dou_em.transpose(Image.ROTATE_90)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.ROTATE_90)

                    elif random_aug == 1:
                        # 旋转180
                        im = im.transpose(Image.ROTATE_180)
                        dou_em = dou_em.transpose(Image.ROTATE_180)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.ROTATE_180)

                    elif random_aug == 2:
                        # 旋转270
                        im = im.transpose(Image.ROTATE_270)
                        dou_em = dou_em.transpose(Image.ROTATE_270)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.ROTATE_270)
                    elif random_aug == 3:
                        # 左右互换
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)
                        dou_em = dou_em.transpose(Image.FLIP_LEFT_RIGHT)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.FLIP_LEFT_RIGHT)
                    elif random_aug == 4:
                        # 左右呼唤
                        im = im.transpose(Image.FLIP_TOP_BOTTOM)
                        dou_em = dou_em.transpose(Image.FLIP_TOP_BOTTOM)
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].transpose(Image.FLIP_TOP_BOTTOM)
            except:
                traceback.print_exc()
        return im, dou_em

    def src_check(self):
        """
        检查src是否有效
        :return:
        """
        pass
    def gt_check(self):
        """
        检查GT是否有效
        :return:
        """
        pass

    def debug_test(self):
        """
        开发的时候用来测
        :return:
        """
        src_path = r'C:\Users\musk\Desktop\tamper_test\Default_78_398895_clock.png'
        gt_path = r'C:\Users\musk\Desktop\tamper_test\Gt_78_398895_clock.bmp'
        self.X_train_or_test = [str(src_path)]
        src = Image.open(src_path)
        gt = Image.open(gt_path)
        relation_8_map = DataAugment.gen_8_2_map(self,np.array(gt))
        for i in range(100):
            im , gt = DataAugment.combine_augment_method(self,src,gt,relation_8_map=relation_8_map,index=0)
            im.save(r'C:\Users\musk\Desktop\tamper_test\augment_test'+'\\'+str(i)+'_src.png')
            gt.save(r'C:\Users\musk\Desktop\tamper_test\augment_test' + '\\' + str(i) + '_gt.bmp')


    def gen_8_2_map(self, mask, mask_area=50, mask_edge=255, not_mask_edge=100, not_mask_area=0, output_type='Image'):
        """
        输入mask，先按照固定参数标好，篡改区域、篡改区域边缘，非篡改区域边缘，非篡改区域
        :param mask: 255 的图，channel数为1
        :return: 从左上角的点开始按照顺时针方向的8张二通道的图
        """
        # 在输入mask 之前对mask进行检查
        if type(mask) is np.ndarray:
            if mask.ndim == 2:
                pass
            else:
                # 如果输入的维度不是3二维的，则转化成2 dim
                print('Notice: when using the function gen_8_2_map, the mask ndim is not 2 but', mask.ndim)
                mask = mask[:,:,0]
        else:
            print('Notice: when using function gen_8_2_map, the input mask not numpy array')
            traceback.print_exc()
            sys.exit()

        # 开始进行8张图的生成
        relation_8_map = []
        edge_loc_ = [1, 1]
        # 找到内侧和外侧边缘
        mask_pad = np.pad(mask, (1, 1), mode='constant')
        mask_pad = np.where(mask_pad == 50, 0, mask_pad)
        edge_loc = np.where(mask_pad == mask_edge)
        edge_loc_1 = np.where(mask_pad == not_mask_edge)
        edge_loc_[0] = np.append(edge_loc[0], edge_loc_1[0])
        edge_loc_[1] = np.append(edge_loc[1], edge_loc_1[1])

        del edge_loc_1
        del edge_loc
        edge_loc = edge_loc_
        mask_shape = mask_pad.shape
        # 生成8张结果图
        for i in range(8):
            temp = np.ones((mask_shape[0], mask_shape[1], 2))
            relation_8_map.append(temp)

        for j in range(len(edge_loc[0])):
            row = edge_loc[0][j]
            col = edge_loc[1][j]
            if mask_pad[row - 1, col - 1] != 0:
                relation_8_map[0][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row - 1, col - 1]:
                    relation_8_map[0][row, col, 1] = 1
                else:
                    relation_8_map[0][row, col, 1] = 0
            else:
                relation_8_map[0][row, col, 0] = 0
                relation_8_map[0][row, col, 1] = 0

            if mask_pad[row - 1, col] != 0:
                relation_8_map[1][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row - 1, col]:
                    relation_8_map[1][row, col, 1] = 1
                else:
                    relation_8_map[1][row, col, 1] = 0
            else:
                relation_8_map[1][row, col, 0] = 0
                relation_8_map[1][row, col, 1] = 0

            if mask_pad[row - 1, col + 1] != 0:
                relation_8_map[2][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row - 1, col + 1]:
                    relation_8_map[2][row, col, 1] = 1
                else:
                    relation_8_map[2][row, col, 1] = 0
            else:
                relation_8_map[2][row, col, 0] = 0
                relation_8_map[2][row, col, 1] = 0

            if mask_pad[row, col + 1] != 0:
                relation_8_map[3][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row, col + 1]:
                    relation_8_map[3][row, col, 1] = 1
                else:
                    relation_8_map[3][row, col, 1] = 0
            else:
                relation_8_map[3][row, col, 0] = 0
                relation_8_map[3][row, col, 1] = 0

            if mask_pad[row + 1, col + 1] != 0:
                relation_8_map[4][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row + 1, col + 1]:
                    relation_8_map[4][row, col, 1] = 1
                else:
                    relation_8_map[4][row, col, 1] = 0
            else:
                relation_8_map[4][row, col, 0] = 0
                relation_8_map[4][row, col, 1] = 0

            if mask_pad[row + 1, col] != 0:
                relation_8_map[5][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row + 1, col]:
                    relation_8_map[5][row, col, 1] = 1
                else:
                    relation_8_map[5][row, col, 1] = 0
            else:
                relation_8_map[5][row, col, 0] = 0
                relation_8_map[5][row, col, 1] = 0

            if mask_pad[row + 1, col - 1] != 0:
                relation_8_map[6][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row + 1, col - 1]:
                    relation_8_map[6][row, col, 1] = 1
                else:
                    relation_8_map[6][row, col, 1] = 0
            else:
                relation_8_map[6][row, col, 0] = 0
                relation_8_map[6][row, col, 1] = 0

            if mask_pad[row, col - 1] != 0:
                relation_8_map[7][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row, col - 1]:
                    relation_8_map[7][row, col, 1] = 1
                else:
                    relation_8_map[7][row, col, 1] = 0
            else:
                relation_8_map[7][row, col, 0] = 0
                relation_8_map[7][row, col, 1] = 0

        for i in range(8):
            relation_8_map[i] = relation_8_map[i][1:-1, 1:-1, :]
            # plt.figure('123')
            # plt.imshow(relation_8_map[i][:,:,0])
            # plt.savefig('C:\\Users\\musk\\Desktop\\边缘8张图关系图\\%d.png'%i)
            # plt.show()
            # # temp = Image.fromarray(relation_8_map[i])
            # # temp = temp.convert('RGB')
            # # temp.save('C:\\Users\\musk\\Desktop\\边缘8张图关系图\\%d.png'%i)
        if output_type == 'Image':
            for i in range(8):
                relation_8_map[i] = Image.fromarray(relation_8_map[i].astype('uint8')).convert('RGB')
        else:
            pass

        return relation_8_map
class MixData():
    def __init__(self):

        # src_path_list = [ '/media/liu/File/10月数据准备/10月12日实验数据/splicing/tamper_result_320',
            # '/media/liu/File/10月数据准备/10月12日实验数据/cm/test_dataset_train_percent_0.80@8_20']
                         # '/media/liu/File/10月数据准备/10月12日实验数据/negative/src',
                         # '/media/liu/File/10月数据准备/10月12日实验数据/casia/src',
        src_path_list = ['/media/liu/File/10月数据准备/10月12日实验数据/cm/test_dataset_train_percent_0.80@8_20',
                           '/media/liu/File/10月数据准备/10月12日实验数据/negative/src',
                        #'/media/liu/File/10月数据准备/10月12日实验数据/splicing/tamper_result_320',
                        # '/media/liu/File/Sp_320_dataset/tamper_result_320',
                        # '/media/liu/File/10月数据准备/10月12日实验数据/casia/src'
                         ]
        gt_path_list = []
        self.src_path_list = src_path_list

        self.sp_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/splicing/ground_truth_result_320'
        # self.sp_gt_path = '/media/liu/File/Sp_320_dataset/ground_truth_result_320'
        # self.sp_gt_path = '/media/liu/File2/ground_truth_result_320'
        self.cm_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/cm/test_gt_train_percent_0.80@8_20'
        self.negative_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'
        self.casia_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/casia/gt'

        # if True:
        #     self.src_path_list = [r'H:\10月数据准备\10月12日实验数据\casia\src']
        #     self.casia_gt_path = r'H:\10月数据准备\10月12日实验数据\casia\gt'

    def gen_dataset(self):
        """
        通过输入的src & gt的路径生成train_list 列表
        并通过check方法，检查是否有误
        :return:
        """
        dataset_type_num = len(self.src_path_list)
        train_list = []
        gt_list = []
        unmatched_list = []
        # 首先开始遍历不同类型的数据集路径
        for index1, item1 in enumerate(self.src_path_list):
            for index2,item2 in enumerate(os.listdir(item1)):
                t_img_path = os.path.join(item1, item2)
                t_gt_path = MixData.__switch_case(self, t_img_path)
                if t_gt_path != '':
                    train_list.append(t_img_path)
                    gt_list.append(t_gt_path)
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

        return train_list, gt_list


    def __check(self, train_list, gt_list):
        """
        检查train_list 和 gt_list 是否有问题
        :return:
        """
        pass

    def __switch_case(self, path):
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
        if len(type) != 1:
            print('The type len is ', len(type))
            return ''

        if type[0] == 'sp':
            gt_path = name.replace('Default','Gt').replace('.jpg','.bmp').replace('.png', '.bmp').replace('poisson','Gt')
            gt_path = os.path.join(self.sp_gt_path, gt_path)
            pass
        elif type[0] == 'cm':
            gt_path = name.replace('Default', 'Gt').replace('.jpg','.bmp').replace('.png', '.bmp').replace('poisson','Gt')
            gt_path = os.path.join(self.cm_gt_path, gt_path)
            pass
        elif type[0] == 'negative':
            gt_path = 'negative_gt.bmp'
            gt_path = os.path.join(self.negative_gt_path, gt_path)
            pass
        elif type[0] == 'casia':
            gt_path = name.split('.')[0] + '_gt' + '.png'
            gt_path = os.path.join(self.casia_gt_path, gt_path)
            pass
        else:
            traceback.print_exc()
            print('Error')
            sys.exit()
        # 判断gt是否存在
        if os.path.exists(gt_path):
            pass
        else:
            return ''

        return gt_path

if __name__ == "__main__":
    # data = DataAugment()
    # data.debug_test()
    # model
    dataParser = DataParser(2)
    batch_ids = np.random.choice(dataParser.X_train, dataParser.batch_size)
    dataParser.get_batch(batch_ids)
    try:
        _ = generate_minibatches(dataParser=dataParser,train=True)
        print()
    except Exception as e:
        traceback.print_exc()