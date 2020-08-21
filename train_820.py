import torch.optim as optim
from functions import my_f1_score, my_acc_score, my_precision_score, weighted_cross_entropy_loss, wce_huber_loss, \
    wce_huber_loss_8
import conf.global_setting as settings
from datasets.dataset import DataParser
from model.model_812 import Net
import os, sys
import numpy as np
from PIL import Image
import shutil
import argparse
import time
import datetime
import torch
from torch.optim import lr_scheduler
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import Logger, Averagvalue, save_checkpoint, weights_init, load_pretrained,save_mid_result
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=6, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=100, type=int, metavar='N',
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/HED')
parser.add_argument('--mid_result_root', type=str, help='mid_result_root', default='./mid_result_820')
parser.add_argument('--model_save_dir', type=str, help='model_save_dir', default='')
parser.add_argument('--mid_result_index',type=list,help='mid_result_index',default=[])
# ================ dataset
parser.add_argument('--dataset', help='root folder of dataset', default='dta/HED-BSD')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

model_save_dir = abspath(dirname(__file__))
model_save_dir = join(model_save_dir, args.model_save_dir)
if not isdir(model_save_dir):
    os.makedirs(model_save_dir)

# tensorboard 使用
writer = SummaryWriter('runs/' + '%d-%d_tensorboard' % (datetime.datetime.now().month, datetime.datetime.now().day))


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
        # ims_t = ims.transpose(0,1,2,3)
        # plt.figure('ims')
        # plt.imshow(ims[0,0,:,:]*255)
        # plt.show()
        #
        # plt.figure('gt')
        # plt.imshow(double_edge[0,0,:,:])
        # plt.show()
        yield (ims, [double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8])


def train(model, optimizer, epoch, save_dir):
    # 读取数据的迭代器
    dataParser = DataParser(args.batch_size)

    #
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()

    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0

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
        else:
            loss = torch.zeros(1)

        # 输出结果[img，8张图]
        outputs = model(images)

        # 这里放保存中间结果的代码
        if batch_index in args.mid_result_index:
            save_mid_result(outputs[0], labels, epoch, batch_index, args.mid_result_root,save_8map=True,train_phase=True)

        # 建立loss
        loss = wce_huber_loss(outputs[0], labels[0]) * 12
        for c_index, c in enumerate(outputs[1:]):
            loss = loss + wce_huber_loss_8(c, labels[c_index + 1])
        loss = loss / 20
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # measure the accuracy and record loss
        losses.update(loss.item(), images.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score = my_f1_score(outputs[0], labels[0])
        precisionscore = my_precision_score(outputs[0], labels[0])
        accscore = my_acc_score(outputs[0], labels[0])
        writer.add_scalar('f1score', f1score, global_step=epoch * dataParser.steps_per_epoch + batch_index)
        writer.add_scalar('precisionscore', precisionscore,
                          global_step=epoch * dataParser.steps_per_epoch + batch_index)
        writer.add_scalar('accscore', accscore, global_step=epoch * dataParser.steps_per_epoch + batch_index)



        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.steps_per_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'f1_score : %.4f ' % f1score + \
                   'precision_score: %.4f ' % precisionscore + \
                   'acc_score %.4f ' % accscore

            print(info)
        writer.add_scalar('tr_avg_loss2', losses.val, global_step=epoch * dataParser.steps_per_epoch + batch_index)
        if batch_index == dataParser.steps_per_epoch:
            break
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))

    return losses.avg, epoch_loss

@torch.no_grad()
def val(model, epoch):
    torch.cuda.empty_cache()
    # 读取数据的迭代器
    dataParser = DataParser(args.batch_size)

    #
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()

    # switch to train mode
    model.eval()
    end = time.time()
    epoch_loss = []
    counter = 0

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

        # 输出结果[img，8张图]
        outputs = model(images)

        # 这里放保存中间结果的代码
        if batch_index in args.mid_result_index:
            save_mid_result(outputs, labels, epoch, batch_index, args.mid_result_root,save_8map=True,train_phase=False)

        # 建立loss
        loss = wce_huber_loss(outputs[0], labels[0]) * 12
        for c_index, c in enumerate(outputs[1:]):
            loss = loss + wce_huber_loss_8(c, labels[c_index + 1])
        loss = loss / 20

        # measure the accuracy and record loss
        losses.update(loss.item(), images.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score = my_f1_score(outputs[0], labels[0])
        precisionscore = my_precision_score(outputs[0], labels[0])
        accscore = my_acc_score(outputs[0], labels[0])
        writer.add_scalar('val_f1score', f1score, global_step=epoch * dataParser.val_steps + batch_index)
        writer.add_scalar('val_precisionscore', precisionscore,
                          global_step=epoch * dataParser.val_steps + batch_index)
        writer.add_scalar('val_acc_score', accscore, global_step=epoch * dataParser.val_steps + batch_index)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.val_steps) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'f1_score : %.4f ' % f1score + \
                   'precision_score: %.4f ' % precisionscore + \
                   'acc_score %.4f ' % accscore

            print('val: ',info)
        writer.add_scalar('val_avg_loss2', losses.val, global_step=epoch * dataParser.steps_per_epoch + batch_index)
        if batch_index == dataParser.val_steps//10:
            break

    return losses.avg, epoch_loss


def main():
    args.cuda = True

    # model
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    # 模型初始化
    # model.apply(weights_init)

    # 模型可持续化
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
            optimizer = optim.SGD(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_dacay)
            optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            optimizer = optim.SGD(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_dacay)
            print("=> no checkpoint found at '{}'".format(args.resume))


    else:
        model = Net()

    # 优化器
    # optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.999),eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    for epoch in range(args.start_epoch, args.maxepoch):
        tr_avg_loss, tr_detail_loss = train(model=model, optimizer=optimizer, epoch=epoch,
                                            save_dir=join(args.model_save_dir, 'epoch-%d-training-record-%d_%d'
                                                          % (epoch, datetime.datetime.now().month,
                                                             datetime.datetime.now().day)))

        val_avg_loss, val_detail_loss = val(model=model, epoch=epoch)
        writer.add_scalar('tr_avg_loss_per_epoch', tr_avg_loss, global_step=epoch)
        writer.add_scalar('lr_per_epoch', scheduler.get_lr(), global_step=epoch)
        writer.add_scalar('val_avg_loss_per_epoch', val_avg_loss, global_step=epoch)

        # 保存模型
        save_file = os.path.join(args.model_save_dir, 'checkpoint_epoch{}.pth'.format(epoch))
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename=save_file)
        scheduler.step()

    print('训练已完成!')



if __name__ == '__main__':
    main()
