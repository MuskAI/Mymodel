import torch
import torch.optim as optim
import numpy as np
import os, sys
import argparse
import time, datetime
from functions import my_f1_score, my_acc_score, my_precision_score, weighted_cross_entropy_loss, wce_huber_loss, \
    wce_huber_loss_8, my_recall_score, debug_ce, cross_entropy_loss, wce_dice_huber_loss
from torch.nn import init
from dataset import DataParser, gen_band_gt
from model.model_stage1_SRM import Net
from PIL import Image
import shutil
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import Logger, Averagvalue, weights_init, load_pretrained, save_mid_result,send_msn
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

"""
Created by HaoRan
time: 11/5
description:
1. stage one training
"""

""""""""""""""""""""""""""""""
"          参数               "
""""""""""""""""""""""""""""""

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=5, type=int, metavar='BT',
                    help='batch size')

# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
'/home/liu/chenhaoran/Mymodel/record823//home/liu/chenhaoran/Mymodel/record823/checkpoint9-stage1-0.002801-f10.790759-precision0.957186-acc0.992177-recall0.685567.pth'

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/HED')
parser.add_argument('--mid_result_root', type=str, help='mid_result_root', default='./mid_result_823')
parser.add_argument('--model_save_dir', type=str, help='model_save_dir', default='./save_model/model_stage_one_srm_band7')
parser.add_argument('--mid_result_index', type=list, help='mid_result_index', default=[0])
parser.add_argument('--per_epoch_freq', type=int, help='per_epoch_freq', default=50)

parser.add_argument('--fuse_loss_weight', type=int, help='fuse_loss_weight', default=12)
# ================ dataset

parser.add_argument('--dataset', help='root folder of dataset', default='dta/HED-BSD')
parser.add_argument('--band_mode', help='weather using band of normal gt', type=bool, default=True)
parser.add_argument('--save_mid_result', help='weather save mid result', type=bool, default=False)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

""""""""""""""""""""""""""""""
"          路径               "
""""""""""""""""""""""""""""""
model_save_dir = abspath(dirname(__file__))
model_save_dir = join(model_save_dir, args.model_save_dir)
if not isdir(model_save_dir):
    os.makedirs(model_save_dir)

# tensorboard 使用
writer = SummaryWriter(
    'runs/' + '1128_band7_srm_%d-%d_tensorboard' % (datetime.datetime.now().month, datetime.datetime.now().day))


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

        # 通道位置转化
        ims = ims.transpose(0, 3, 1, 2)
        chanel1 = chanel1.transpose(0, 3, 1, 2)
        chanel2 = chanel2.transpose(0, 3, 1, 2)
        chanel3 = chanel3.transpose(0, 3, 1, 2)
        chanel4 = chanel4.transpose(0, 3, 1, 2)
        chanel5 = chanel5.transpose(0, 3, 1, 2)
        chanel6 = chanel6.transpose(0, 3, 1, 2)
        chanel7 = chanel7.transpose(0, 3, 1, 2)
        chanel8 = chanel8.transpose(0, 3, 1, 2)
        double_edge = double_edge.transpose(0, 3, 1, 2)

        # 设置是否要使用条带
        if True:
            double_edge = gen_band_gt(double_edge)
            pass
        # ims_t = ims.transpose(0,1,2,3)
        # plt.figure('ims')
        # plt.imshow(ims[0,0,:,:]*255)
        # plt.show()
        #
        # # plt.show()
        # # plt.savefig("temp_ims.png")
        #
        # plt.figure('gt')
        # plt.imshow(double_edge[0,0,:,:])
        # plt.show()

        # plt.show()
        # plt.savefig("temp_gt.png")
        yield (ims, [double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8])


""""""""""""""""""""""""""""""
"          程序入口            "
""""""""""""""""""""""""""""""


def main():
    args.cuda = True
    # data
    dataParser = DataParser(args.batch_size)
    # model
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    model.apply(weights_init)
    # 模型初始化
    # 如果没有这一步会根据正态分布自动初始化
    # model.apply(weights_init)

    # 模型可持续化
    # 这是tensorflow代码中的配置：    optimizer = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(args.resume))
            # optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        print('start learning')
    # 调整学习率
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # 数据迭代器

    for epoch in range(args.start_epoch, args.maxepoch):
        train_avg = train(model=model, optimizer=optimizer, dataParser=dataParser, epoch=epoch)
        val_avg = val(model=model, dataParser=dataParser, epoch=epoch)

        """"""""""""""""""""""""""""""
        "          写入图            "
        """"""""""""""""""""""""""""""
        writer.add_scalar('tr_avg_loss_per_epoch', train_avg['loss_avg'], global_step=epoch)
        writer.add_scalar('tr_avg_f1_per_epoch', train_avg['f1_avg'], global_step=epoch)
        writer.add_scalar('tr_avg_precision_per_epoch', train_avg['precision_avg'], global_step=epoch)
        writer.add_scalar('tr_avg_acc_per_epoch', train_avg['accuracy_avg'], global_step=epoch)
        writer.add_scalar('tr_avg_recall_per_epoch', train_avg['recall_avg'], global_step=epoch)

        writer.add_scalar('val_avg_loss_per_epoch', val_avg['loss_avg'], global_step=epoch)
        writer.add_scalar('val_avg_f1_per_epoch', val_avg['f1_avg'], global_step=epoch)
        writer.add_scalar('val_avg_precision_per_epoch', val_avg['precision_avg'], global_step=epoch)
        writer.add_scalar('val_avg_acc_per_epoch', val_avg['accuracy_avg'], global_step=epoch)
        writer.add_scalar('val_avg_recall_per_epoch', val_avg['recall_avg'], global_step=epoch)

        writer.add_scalar('lr_per_epoch', scheduler.get_lr(), global_step=epoch)

        """"""""""""""""""""""""""""""
        "          写入图            "
        """"""""""""""""""""""""""""""

        # 保存模型
        """
                    info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.steps_per_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
                   'precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value) + \
                   'acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) +\
                   'recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)

        """
        output_name = '1121checkpoint%d-stage1-%f-f1%f-precision%f-acc%f-recall%f.pth' % (epoch,val_avg['loss_avg'],val_avg['f1_avg'],
                                                                                                      val_avg['precision_avg'],
                                                                                                      val_avg['accuracy_avg'],
                                                                                                      val_avg['recall_avg'])
        try:
            send_msn(epoch,f1=val_avg['f1_avg'])
        except:
            pass
        if epoch % 1 == 0:
            save_model_name = os.path.join(args.model_save_dir, output_name)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       save_model_name)

        scheduler.step(epoch)

    print('训练已完成!')


""""""""""""""""""""""""""""""
"           训练              "
""""""""""""""""""""""""""""""


def train(model, optimizer, dataParser, epoch):
    # 读取数据的迭代器
    train_epoch = int(dataParser.steps_per_epoch)
    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    f1_value = Averagvalue()
    acc_value = Averagvalue()
    recall_value = Averagvalue()
    precision_value = Averagvalue()
    map8_loss_value = Averagvalue()

    # switch to train mode
    model.train()
    end = time.time()

    for batch_index, (images, labels_numpy) in enumerate(generate_minibatches(dataParser, True)):
        # 读取数据的时间
        data_time.update(time.time() - end)

        # 对读取的numpy类型数据进行调整
        labels = []
        if torch.cuda.is_available():
            images = torch.from_numpy(images).cuda()
            for item in labels_numpy:
                labels.append(torch.from_numpy(item).cuda())
        else:
            images = torch.from_numpy(images)
            for item in labels_numpy:
                labels.append(torch.from_numpy(item))

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
            loss_8t = torch.zeros(()).cuda()
        else:
            loss = torch.zeros(1)
            loss_8t = torch.zeros(())

        with torch.set_grad_enabled(True):
            images.requires_grad = True
            optimizer.zero_grad()
            # 网络输出
            outputs = model(images)
            # 这里放保存中间结果的代码
            if args.save_mid_result:
                if batch_index in args.mid_result_index:
                    save_mid_result(outputs, labels, epoch, batch_index, args.mid_result_root, save_8map=True,
                                    train_phase=True)
                else:
                    pass
            else:
                pass
            """"""""""""""""""""""""""""""
            "         Loss 函数           "
            """"""""""""""""""""""""""""""

            if not args.band_mode:
                # 如果不是使用band_mode 则需要计算8张图的loss
                loss = wce_dice_huber_loss(outputs[0], labels[0]) * args.fuse_loss_weight

                writer.add_scalar('fuse_loss_per_epoch', loss.item() / args.fuse_loss_weight,
                                  global_step=epoch * train_epoch + batch_index)

                for c_index, c in enumerate(outputs[1:]):
                    one_loss_t = wce_dice_huber_loss(c, labels[c_index + 1])
                    loss_8t += one_loss_t
                    writer.add_scalar('%d_map_loss' % (c_index), one_loss_t.item(), global_step=train_epoch)
                loss += loss_8t
                loss = loss / 20
            else:
                loss = wce_dice_huber_loss(outputs[0], labels[0])
                writer.add_scalar('fuse_loss_per_epoch', loss.item(),
                                  global_step=epoch * train_epoch + batch_index)

            loss.backward()
            optimizer.step()

        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        map8_loss_value.update(loss_8t.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score = my_f1_score(outputs[0], labels[0])
        precisionscore = my_precision_score(outputs[0], labels[0])
        accscore = my_acc_score(outputs[0], labels[0])
        recallscore = my_recall_score(outputs[0], labels[0])

        writer.add_scalar('f1_score', f1score, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('precision_score', precisionscore, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('acc_score', accscore, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('recall_score', recallscore, global_step=epoch * train_epoch + batch_index)
        ################################

        f1_value.update(f1score)
        precision_value.update(precisionscore)
        acc_value.update(accscore)
        recall_value.update(recallscore)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.steps_per_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
                   'precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value) + \
                   'acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) + \
                   'recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)

            print(info)

        if batch_index >= train_epoch:
            break

    return {'loss_avg': losses.avg,
            'f1_avg': f1_value.avg,
            'precision_avg': precision_value.avg,
            'accuracy_avg': acc_value.avg,
            'recall_avg': recall_value.avg}


@torch.no_grad()
def val(model, dataParser, epoch):
    # 读取数据的迭代器
    train_epoch = int(dataParser.val_steps)
    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    f1_value = Averagvalue()
    acc_value = Averagvalue()
    recall_value = Averagvalue()
    precision_value = Averagvalue()
    map8_loss_value = Averagvalue()

    # switch to test mode
    model.eval()
    end = time.time()

    for batch_index, (images, labels_numpy) in enumerate(generate_minibatches(dataParser, False)):
        # 读取数据的时间
        data_time.update(time.time() - end)

        # 对读取的numpy类型数据进行调整
        labels = []
        if torch.cuda.is_available():
            images = torch.from_numpy(images).cuda()
            for item in labels_numpy:
                labels.append(torch.from_numpy(item).cuda())
        else:
            images = torch.from_numpy(images)
            for item in labels_numpy:
                labels.append(torch.from_numpy(item))

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
            loss_8t = torch.zeros(()).cuda()
        else:
            loss = torch.zeros(1)
            loss_8t = torch.zeros(())

        # 网络输出
        outputs = model(images)
        # 这里放保存中间结果的代码
        if args.save_mid_result:
            if batch_index in args.mid_result_index:
                save_mid_result(outputs, labels, epoch, batch_index, args.mid_result_root, save_8map=True,
                                train_phase=True)
            else:
                pass
        else:
            pass
        """"""""""""""""""""""""""""""
        "         Loss 函数           "
        """"""""""""""""""""""""""""""

        if not args.band_mode:
            # 如果不是使用band_mode 则需要计算8张图的loss
            loss = wce_dice_huber_loss(outputs[0], labels[0]) * args.fuse_loss_weight

            writer.add_scalar('val_fuse_loss_per_epoch', loss.item() / args.fuse_loss_weight,
                              global_step=epoch * train_epoch + batch_index)

            for c_index, c in enumerate(outputs[1:]):
                one_loss_t = wce_dice_huber_loss(c, labels[c_index + 1])
                loss_8t += one_loss_t
                writer.add_scalar('val_%d_map_loss' % (c_index), one_loss_t.item(), global_step=train_epoch)
            loss += loss_8t
            loss = loss / 20
        else:
            loss = wce_dice_huber_loss(outputs[0], labels[0])
            writer.add_scalar('val_fuse_loss_per_epoch', loss.item(),
                              global_step=epoch * train_epoch + batch_index)

        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        map8_loss_value.update(loss_8t.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score = my_f1_score(outputs[0], labels[0])
        precisionscore = my_precision_score(outputs[0], labels[0])
        accscore = my_acc_score(outputs[0], labels[0])
        recallscore = my_recall_score(outputs[0], labels[0])

        writer.add_scalar('val_f1_score', f1score, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('val_precision_score', precisionscore, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('val_acc_score', accscore, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('val_recall_score', recallscore, global_step=epoch * train_epoch + batch_index)
        ################################

        f1_value.update(f1score)
        precision_value.update(precisionscore)
        acc_value.update(accscore)
        recall_value.update(recallscore)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.val_steps) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'vla_Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'val_f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
                   'val_precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value) + \
                   'val_acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) + \
                   'val_recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)

            print(info)

        if batch_index >= train_epoch:
            break

    return {'loss_avg': losses.avg,
            'f1_avg': f1_value.avg,
            'precision_avg': precision_value.avg,
            'accuracy_avg': acc_value.avg,
            'recall_avg': recall_value.avg}


if __name__ == '__main__':
    main()
