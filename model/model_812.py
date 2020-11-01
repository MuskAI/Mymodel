import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary
# from tensorboardX import SummaryWriter


# writer = SummaryWriter('runs/')

class ResBlock(nn.Module):
    """
    残差模块
       self.res_block1 = ResBlock(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=2,
                                   kernel_size=3)
    """

    def __init__(self, in_channel, out_channel, mid_channel1, mid_channel2, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_channel1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=mid_channel1, out_channels=mid_channel2, kernel_size=kernel_size, stride=1,
                               padding_mode='replicate', padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(mid_channel2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=mid_channel2, out_channels=out_channel, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class Aspp(nn.Module):
    def __init__(self, input_shape, out_stride):
        super(Aspp, self).__init__()
        self.out_shape = int(input_shape[0] / out_stride)
        self.out_shape1 = int(input_shape[1] / out_stride)

        self.b0 = nn.Sequential(OrderedDict([
            ('b0_conv', nn.Conv2d(256, 128, kernel_size=1, padding_mode='replicate', bias=False)),
            ('b0_bn', nn.BatchNorm2d(128)),
            ('b0_relu', nn.ReLU(inplace=True))
        ]))
        # 可分离卷积
        self.b1 = nn.Sequential(OrderedDict([
            ('b1_depthwise_conv',
             nn.Conv2d(256, 256, kernel_size=3, groups=256, dilation=(6, 6), padding_mode='replicate', padding=6,
                       bias=False)),
            ('b1_bn', nn.BatchNorm2d(256)),
            ('b1_relu', nn.ReLU(inplace=True)),
            ('b1_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b1_bn', nn.BatchNorm2d(256)),
            ('b1_relu', nn.ReLU(inplace=True))
        ]))

        # 又是一个可分离卷积

        self.b2 = nn.Sequential(OrderedDict([
            ('b2_depthwise_conv',
             nn.Conv2d(256, 256, kernel_size=3, groups=256, dilation=(12, 12), padding_mode='replicate', padding=12,
                       bias=False)),
            ('b2_bn', nn.BatchNorm2d(256)),
            ('b2_relu', nn.ReLU(inplace=True)),
            ('b2_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b2_bn', nn.BatchNorm2d(256)),
            ('b2_relu', nn.ReLU(inplace=True))
        ]))

        self.b3 = nn.Sequential(OrderedDict([
            ('b3_depthwise_conv',
             nn.Conv2d(256, 256, kernel_size=3, groups=256, dilation=(12, 12), padding_mode='replicate', padding=12,
                       bias=False)),
            ('b3_bn', nn.BatchNorm2d(256)),
            ('b3_relu', nn.ReLU(inplace=True)),
            ('b3_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b3_bn', nn.BatchNorm2d(256)),
            ('b3_relu', nn.ReLU(inplace=True))
        ]))
        # self.b4_ = nn.AdaptiveAvgPool2d((1, 1))
        self.b4 = nn.Sequential(OrderedDict([
            ('b4_averagepool', nn.AdaptiveAvgPool2d((1,1))),
            ('b4_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b4_bn', nn.BatchNorm2d(128)),
            ('b4_relu', nn.ReLU(inplace=True)),
            ('b4_bilinearUpsampling',nn.UpsamplingBilinear2d(size=(self.out_shape, self.out_shape1), scale_factor=None))
        ]))


    def forward(self, x):
        b0_ = self.b0(x)
        b1_ = self.b1(x)
        b2_ = self.b2(x)
        b3_ = self.b3(x)
        # b4_ = self.b4(x)
        b4_ = self.b4(x)
        x = torch.cat([b4_, b0_, b1_, b2_, b3_], dim=1)
        return x


class Net(nn.Module):
    def __init__(self, input_shape=(320, 320, 3)):
        super(Net, self).__init__()
        self.input_shape = input_shape

        # Step1: 对输入的图进行处理
        self.in_conv_bn_relu = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding_mode='replicate', padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #######################################

        # Step2:第一个和第二个res_block + shortcut add

        self.res_block1 = ResBlock(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=1,
                                   kernel_size=3)
        self.res_block1_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.res_block2 = ResBlock(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=1,
                                   kernel_size=3)

        #####################################

        # Step3:

        self.res_block3 = ResBlock(in_channel=64, out_channel=128, mid_channel1=64, mid_channel2=64, stride=1,
                                   kernel_size=3)
        self.res_block3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        self.res_block4 = ResBlock(in_channel=128, out_channel=128, mid_channel1=64, mid_channel2=64, kernel_size=3)
        self.res_block5 = ResBlock(in_channel=128, out_channel=128, mid_channel1=64, mid_channel2=64, kernel_size=3)

        ####################################

        # Step4:

        self.res_block6 = ResBlock(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=2)
        self.res_block6_shortcut = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )

        self.res_block7 = ResBlock(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=1)
        self.res_block8 = ResBlock(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=1)
        self.res_block9 = ResBlock(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=1)
        # 下一个
        self.res_block10 = ResBlock(in_channel=128, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=2)
        self.res_block10_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1,stride=2),
            nn.BatchNorm2d(256)
        )

        self.res_block11 = ResBlock(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block12 = ResBlock(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block13 = ResBlock(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block14 = ResBlock(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block15 = ResBlock(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)

        #####################################

        # Step5: aspp上面的几个模块
        self.res_block16 = ResBlock(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=2)
        self.res_block16_shortcut = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1,stride=2),
            nn.BatchNorm2d(256)
        )

        self.res_block17 = ResBlock(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block18 = ResBlock(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)

        ######################################

        # Step6:Aspp和它下面的部分
        self.aspp = Aspp(input_shape=(320, 320, 3), out_stride=16)

        ####################################

        # Step7:开始上采样部分
        ## 20--->40部分

        self.aspp_below_20_40 = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip40_40 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        self.skip80_40 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.aspp_shortcut_40_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 40--->80解码器部分
        self.up40_80 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip80_80 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip160_80 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up40_80_shortcut_after_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 80--->160解码器部分
        self.up80_160 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip160_160_l = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip160_160_r = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip320_320 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up80_160_shortcut_after_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 160--->320解码器部分
        self.up160_320 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up160_320_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_final = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        ###################################

        # 结束上采样部分，开始做最后输出的准备

        ## 8张图
        self.relation_map_1 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )

        self.relation_map_2 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_3 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_4 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_5 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_6 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_7 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_8 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        ## 8张图旁边的skip
        self.relation_map_skip = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Sigmoid()
        )
        ##################################

        # 最后的输出
        self.fusion_out = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_conv_bn_relu(x)
        res_block = self.res_block1(x)
        res_block_shortcut = self.res_block1_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block2(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        # 这里的分支有点复杂，仔细一点

        ######### 320--->320 尺寸的跳跃连接
        skip_320_320 = self.skip320_320(x)
        #########################

        ## 左分支，MaxPooling2D,这里通过计算p=0.5，所以p = 1但是还需要验证一下
        maxpool_down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        ######### 160--->160尺寸的跳跃连接
        skip_160_160_r = self.skip160_160_r(maxpool_down)
        #########################

        res_block = self.res_block3(maxpool_down)
        res_block_shortcut = self.res_block3_shortcut(maxpool_down)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block4(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block5(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        ##### 160--->80尺度的跳远连接
        skip_160_80 = self.skip160_80(x)
        ###########################
        dropout = nn.Dropout2d(0.5)(x)
        ## 160--->160的跳跃连接
        skip_160_160_l = self.skip160_160_l(dropout)
        skip_160_shortcut = torch.cat([skip_160_160_l, skip_160_160_r], 1)
        #######################

        res_block = self.res_block6(dropout)
        res_block_shortcut = self.res_block6_shortcut(dropout)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block7(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block8(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block9(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)
        ## 80--->40 skip
        skip_80_40 = self.skip80_40(x)
        ##############################
        ## 第二个dropout的地方
        dropout = nn.Dropout2d(0.5)(x)
        #### 80 --->80尺度的skip
        skip_80_80 = self.skip80_80(dropout)
        skip_80_shortcut = torch.cat([skip_160_80, skip_80_80], 1)
        #######################

        ## 左分支
        res_block = self.res_block10(x)
        res_block_shortcut = self.res_block10_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block11(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block12(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block13(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block14(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block15(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        ## 第三个dropout的地方
        x = nn.Dropout2d(0.5)(x)
        ## 40--->40尺度的skip
        skip_40_40 = self.skip40_40(x)

        skip_40_shortcut = torch.cat([skip_40_40, skip_80_40], 1)
        #######################
        res_block = self.res_block16(x)
        res_block_shortcut = self.res_block16_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)

        res_block = self.res_block17(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block18(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.Dropout2d(0.5)(x)

        x = self.aspp(x)
        x = self.aspp_below_20_40(x)
        skip = self.aspp_shortcut_40_cat(skip_40_shortcut)
        up1 = torch.cat([x, skip], 1)
        up1 = self.up40_80(up1)

        skip = self.up40_80_shortcut_after_cat(skip_80_shortcut)
        up2 = torch.cat([up1, skip], 1)
        up2 = self.up80_160(up2)

        skip = self.up80_160_shortcut_after_cat(skip_160_shortcut)
        up3 = torch.cat([up2, skip], 1)
        up3 = self.up160_320(up3)

        x = torch.cat([up3, skip_320_320], 1)

        x = self.up_final(x)
        relation_map_1 = self.relation_map_1(x)
        relation_map_2 = self.relation_map_2(x)
        relation_map_3 = self.relation_map_3(x)
        relation_map_4 = self.relation_map_4(x)

        relation_map_5 = self.relation_map_5(x)
        relation_map_6 = self.relation_map_6(x)
        relation_map_7 = self.relation_map_7(x)
        relation_map_8 = self.relation_map_8(x)

        relation_map_skip = self.relation_map_skip(x)

        x = torch.cat([relation_map_1, relation_map_2, relation_map_3, relation_map_4, relation_map_5, relation_map_6,
                       relation_map_7, relation_map_8, relation_map_skip], 1)
        x = self.fusion_out(x)
        #################################
        return [x,relation_map_1, relation_map_2, relation_map_3, relation_map_4, relation_map_5, relation_map_6,
                   relation_map_7, relation_map_8]
        # return [relation_map_1, relation_map_1, relation_map_1, relation_map_1, relation_map_1, relation_map_1, relation_map_1, relation_map_1, x]

if __name__ == '__main__':
    xx = torch.rand(2,3,320,320).cuda()
    model = Net().cuda()
    writer.add_graph(model,xx)
    summary(model,(3,320,320))
    print('ok')