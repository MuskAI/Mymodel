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

from functions import sigmoid_cross_entropy_loss, cross_entropy_loss,l1_loss
from utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=2, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/HED')
# ================ dataset
parser.add_argument('--dataset', help='root folder of dataset', default='data/HED-BSDS')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp)
if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

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
        ims = ims.transpose(0,3,1,2)
        chanel1 = chanel1.transpose(0,3,1,2)
        chanel2 = chanel2.transpose(0, 3, 1, 2)
        chanel3 = chanel3.transpose(0, 3, 1, 2)
        chanel4 = chanel4.transpose(0, 3, 1, 2)
        chanel5 = chanel5.transpose(0, 3, 1, 2)
        chanel6 = chanel6.transpose(0, 3, 1, 2)
        chanel7 = chanel7.transpose(0, 3, 1, 2)
        chanel8 = chanel8.transpose(0, 3, 1, 2)
        double_edge = double_edge.transpose(0,3,1,2)
        yield(ims,[chanel1,chanel2,chanel3,chanel4,chanel5,chanel6,chanel7,chanel8,double_edge])


def train(model,optimizer,epoch,save_dir):
    dataParser = DataParser(args.batch_size)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0


    for batch_index ,(images,labels_numpy) in enumerate(generate_minibatches(dataParser,True)):

        # measure data loading time
        data_time.update(time.time()-end)

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
            loss =torch.zeros(1).cuda()
        else:
            loss = torch.zeros(1)

        optimizer.zero_grad()
        outputs = model(images)
        # 四张GT监督

        for o in outputs[9:]: # o2 o3 o4
            t_loss = cross_entropy_loss(o, labels[-1])
            loss = loss +t_loss
        counter +=1

        for c_index,c in enumerate(outputs[:8]):
            loss = loss + cross_entropy_loss(c, labels[c_index])
        loss = loss/11
        loss.backward()
        acc_scroe = my_accuracy_score(outputs[9].cpu().detach().numpy(),labels[-1].cpu().detach().numpy())
        print('the acc is :',acc_scroe)




        # 下面应该是用来解决batch size 过下的问题
        # if counter == args.itersize:
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     counter = 0

        optimizer.step()
        optimizer.zero_grad()

        # measure the accuracy and record loss
        losses.update(loss.item(),images.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time()-end)
        end = time.time()

        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if batch_index % args.print_freq ==0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.steps_per_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)

        # torch.save(model,join(save_dir,"checkpoint.pth"))
    # 每一轮保存一次参数
    save_checkpoint({'epoch': epoch,'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict()},filename=join(save_dir,"epooch-%d-checkpoint.pth" %epoch))


    return losses.avg,epoch_loss



def test():
    pass


def main():
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    else:
        pass
    model.apply(weights_init)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 数据处理
    # 直接在train里面处理
    # dataParser = DataParser(batch_size)
    loss_function = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=settings.MILESTONES,gamma=0.2)#learning rate decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('Adam', args.lr)))
    sys.stdout = log
    train_loss = []
    train_loss_detail = []

    for epoch in range(args.start_epoch, args.maxepoch):
        if epoch == 0:
            print("Performing initial testing...")
            # 暂时空着

        tr_avg_loss, tr_detail_loss = train(model = model,optimizer = optimizer,epoch= epoch,save_dir=join(TMP_DIR, 'epoch-%d-training-record' % epoch))
        test()

        log.flush()
        # Save checkpoint
        save_file = os.path.join(TMP_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()})

        scheduler.step()  # 自动调整学习率
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss


if __name__ == '__main__':
    main()