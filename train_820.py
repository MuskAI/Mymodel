import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from functions import my_f1_score,my_acc_score,my_precision_score,weighted_cross_entropy_loss
from functions import my_accuracy_score,my_precision_score,weighted_cross_entropy_loss,wce_huber_loss,wce_huber_loss_8
import conf.global_setting as settings
#from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
from datasets.dataset import DataParser
from model.model_812 import Net
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
from torch.optim import lr_scheduler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from functions import sigmoid_cross_entropy_loss, cross_entropy_loss,l1_loss,CE_loss,smooth_l1_loss
from utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain,weights_init
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
parser.add_argument('--resume', default=False, type=bool, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/HED')
parser.add_argument('--mid_result_root',type=str, help='mid_result_root', default='./mid_result_820')
parser.add_argument('--model_save_dir',type=str, help='model_save_dir', default='')
# ================ dataset
parser.add_argument('--dataset', help='root folder of dataset', default='dta/HED-BSD')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

model_save_dir = abspath(dirname(__file__))
model_save_dir = join(model_save_dir, args.model_save_dir)
if not isdir(model_save_dir):
  os.makedirs(model_save_dir)

# tensorboard 使用
writer = SummaryWriter('runs/'+'%d-%d_tensorboard'%(datetime.datetime.now().month,datetime.datetime.now().day))

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
        # ims_t = ims.transpose(0,1,2,3)
        # plt.figure('ims')
        # plt.imshow(ims[0,0,:,:]*255)
        # plt.show()
        #
        # plt.figure('gt')
        # plt.imshow(double_edge[0,0,:,:])
        # plt.show()
        yield(ims,[double_edge,chanel1,chanel2,chanel3,chanel4,chanel5,chanel6,chanel7,chanel8])


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
        # 这里放保存中间结果的代码
        if batch_index in [0,1000,2000,3000]:
            save_mid_result(outputs[0],labels[0],epoch,batch_index,args.mid_result_root)


        loss =wce_huber_loss(outputs[0],labels[0])*12

        for c_index,c in enumerate(outputs[1:]):
            loss = loss + wce_huber_loss_8(c, labels[c_index+1])
        loss = loss/20
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 评价指标
        f1score = my_f1_score(outputs[0],labels[0])
        precisionscore = my_precision_score(outputs[0],labels[0])
        accscore = my_acc_score(outputs[0],labels[0])
        writer.add_scalar('f1score',f1score,global_step=epoch*dataParser.steps_per_epoch+batch_index)
        writer.add_scalar('precisionscore', precisionscore, global_step=epoch * dataParser.steps_per_epoch + batch_index)
        writer.add_scalar('accscore', accscore, global_step=epoch * dataParser.steps_per_epoch + batch_index)



        # measure the accuracy and record loss
        losses.update(loss.item(),images.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time()-end)
        end = time.time()


        if batch_index % 5 ==0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.steps_per_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses)+\
                   'f1_score : %.4f '%f1score +\
                   'precision_score: %.4f '%precisionscore+\
                   'acc_score %.4f '%accscore

            print(info)
        writer.add_scalar('tr_avg_loss2', losses.val, global_step=epoch * dataParser.steps_per_epoch + batch_index)
        if batch_index == dataParser.steps_per_epoch:
            break
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename=save_dir)

    return losses.avg,epoch_loss



def val(model,epoch):
    dataParser = DataParser(args.batch_size)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.eval()
    end = time.time()
    epoch_loss = []
    counter = 0

    for batch_index ,(images,labels_numpy) in enumerate(generate_minibatches(dataParser,False)):

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


        outputs = model(images)
        # 四张GT监督
        loss = wce_huber_loss(outputs[0],labels[0])*12
        # for o in outputs[9:]: # o2 o3 o4
        #     t_loss = cross_entropy_loss(o, labels[-1])
        #     loss = loss +t_loss
        # counter +=1

        for c_index,c in enumerate(outputs[1:]):
            loss = loss + wce_huber_loss_8(c, labels[c_index+1])
        loss = loss/20

        # acc_scroe = my_accuracy_score(outputs[9].cpu().detach().numpy(),labels[-1].cpu().detach().numpy())
        # print('the acc is :',acc_scroe)

        # measure the accuracy and record loss
        losses.update(loss.item(),images.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time()-end)
        end = time.time()

        if batch_index % 5 ==0:
            info = 'val:'+'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.steps_per_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)

            print(info)
        if batch_index == dataParser.steps_per_epoch//10:
            break

    return losses.avg,epoch_loss

def main():
    # 模型可持续化
    if args.resume:
        save_model_path = os.listdir('./record/')
        if save_model_path !=[]:
            model = torch.load('./record'+ '/'+'epoch-6-training-record.pth')
            print('成功加载模型','./record'+ '/'+'epoch-6-training-record.pth')
    else:
        model = Net()
        model.apply(weights_init)

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.999),eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    train_loss = []
    train_loss_detail = []

    for epoch in range(args.start_epoch, args.maxepoch):
        if epoch == 0:
            print("Performing initial testing...")
            # 暂时空着

        tr_avg_loss, tr_detail_loss = train(model = model,optimizer = optimizer,epoch= epoch,save_dir=join(args.model_save_dir, 'epoch-%d-training-record' % epoch))

        # val_avg_loss, val_detail_loss = val(model=model,epoch=epoch)
        writer.add_scalar('tr_avg_loss', tr_avg_loss, global_step=epoch)
        writer.add_scalar('lr', scheduler.get_lr(), global_step=epoch)
        # writer.add_scalar('val_avg_loss', val_avg_loss, global_step=epoch)
        # log.flush()
        # Save checkpoint
        save_file = os.path.join(args.model_save_dir, 'checkpoint_epoch{}.pth'.format(epoch))
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},filename=save_file)
        scheduler.step()

        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss

def save_mid_result(mid_output,label,epoch,batch_index,mid_save_root='./mid_result_820'):
    """
    输入一个batch的gpu tensor,保存中间结果
    :param mid_output:
    :return:
    """
    if mid_save_root == '':
        print('mid_save_root为空')
        sys.exit()

    if os.path.exists(mid_save_root) == False:
        print('中间结果根目录不存在，正在创建')
        os.mkdir(mid_save_root)
    else:
        pass


    dir_name = 'mid_result_epoch:%d'%(epoch)
    dir_path = os.path.join(mid_save_root,dir_name)
    mid_output_dir = os.path.join(dir_path, 'mid_output')
    mid_label_dir = os.path.join(dir_path, 'mid_label')
    if os.path.exists(dir_path) ==False:
        os.mkdir(dir_path)
        os.mkdir(mid_output_dir)
        os.mkdir(mid_label_dir)
        print(dir_path)
        print(mid_label_dir)
        print(mid_output_dir)
        print('创建目录成功！！！！！')
    else:
        print(dir_path,'已经存在')


    for index in range(len(mid_output)):
        file_name_output = 'mid_output_epoch%d_batch_index%d@%d.png' % (epoch, batch_index, index)
        file_output_dir = os.path.join(mid_output_dir,file_name_output)
        file_name_label = 'mid_label_epoch%d_batch_index%d@%d.png' % (epoch, batch_index, index)
        file_label_dir = os.path.join(mid_label_dir,file_name_label)

        show_outputs = np.array(mid_output[index].cpu().detach()) * 255
        show_outputs = np.array(show_outputs, dtype='uint8')
        show_outputs = Image.fromarray(show_outputs[0, 0, :, :]).convert('RGB')
        show_outputs.save(file_output_dir)

        show_labels = np.array(label[index].cpu().detach()) * 255
        show_labels = np.array(show_labels, dtype='uint8')
        show_labels = Image.fromarray(show_labels[0, 0, :, :]).convert('RGB')
        show_labels.save(file_label_dir)


if __name__ == '__main__':
    main()