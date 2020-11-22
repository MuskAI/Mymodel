"""
created by haoran
time:11/19
description:
1. freeze stage1 to train stage2
2.
"""
import torch.optim as optim
from functions import my_f1_score, my_acc_score, my_precision_score, weighted_cross_entropy_loss, wce_dice_huber_loss, my_recall_score,wce_dice_huber_loss
from torch.nn import init
from datasets.dataset import DataParser

from datasets.dataset import gen_band_gt
from model.model_two_stage import Net_Stage_1 as Net1
from model.model_two_stage import Net_Stage_2 as Net2
import os, sys
import numpy as np
from PIL import Image
import shutil
import argparse
import time
import datetime
import torch
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import Logger, Averagvalue, weights_init, load_pretrained,save_mid_result,send_msn
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
""""""""""""""""""""""""""""""""""""
"             参数                  "
""""""""""""""""""""""""""""""""""""
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=3, type=int, metavar='BT',
                    help='batch size')

# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=10, type=int,
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/HED')
parser.add_argument('--mid_result_root', type=str, help='mid_result_root', default='./mid_result_823')
parser.add_argument('--model_save_dir', type=str, help='model_save_dir', default='./save_model119')
parser.add_argument('--mid_result_index',type=list,help='mid_result_index',default=[0])
parser.add_argument('--per_epoch_freq',type=int,help='per_epoch_freq',default=50)
# ================ dataset

parser.add_argument('--dataset', help='root folder of dataset', default='dta/HED-BSD')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
""""""""""""""""""""""""""""""""""""
"            路径准备               "
""""""""""""""""""""""""""""""""""""
model_save_dir = abspath(dirname(__file__))
model_save_dir = join(model_save_dir, args.model_save_dir)
if not isdir(model_save_dir):
    os.makedirs(model_save_dir)

# tensorboard 使用
writer = SummaryWriter('runs/' + '1119_%d-%d_tensorboard' % (datetime.datetime.now().month, datetime.datetime.now().day))
two_stage_input_path = 'mid_result1119/two_stage_input'
one_stage_input_path = 'mid_result1119/one_stage_input'
two_stage_output_path = 'mid_result1119/two_stage_output'
one_stage_output_path = 'mid_result1119/one_stage_output'
_t_list = [two_stage_input_path,one_stage_output_path,one_stage_input_path,two_stage_output_path]
for i in _t_list:
    if os.path.exists(os.path.join('/home/liu/chenhaoran/Mymodel',i)):
        pass
    else:
        os.mkdir(os.path.join('/home/liu/chenhaoran/Mymodel',i))
""""""""""""""""""""""""""""""""""""
"            数据读取               "
""""""""""""""""""""""""""""""""""""


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
        _double_edge = double_edge.copy()
        band_gt = gen_band_gt(_double_edge)
        yield (ims, [double_edge,band_gt, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8])



def main():
    args.cuda = True

    # model
    model1 = Net1().eval()
    model2 = Net2()
    #############################

    if torch.cuda.is_available():
        model1.cuda().eval()
        model2.cuda()
    else:
        model1.cpu().eval()
        model2.cpu()

    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)


    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model1.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
            # optimizer1.load_state_dict(checkpoint['optimizer'])

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


    if True:
        checkpoint1 = torch.load('/home/liu/chenhaoran/Mymodel/record823/1111checkpoint8-stage1-0.296349-f10.863817-precision0.943144-acc0.995090-recall0.803569.pth')
        model1.load_state_dict(checkpoint1['state_dict'])
        checkpoint2 = torch.load(
            '/home/liu/chenhaoran/Mymodel/save_model119/1119checkpoint0-stage2-0.237375-f10.716162-precision0.938563-acc0.994791-recall0.597392.pth')
        model2.load_state_dict(checkpoint2['state_dict'])

        # model.load_state_dict(torch.load('/home/liu/chenhaoran/Mymodel/record823/1031checkpoint_epoch6.pth'))
        print('load sucess')


    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=3, gamma=args.gamma)

    # 数据迭代器
    dataParser = DataParser(args.batch_size)
    for epoch in range(args.start_epoch, args.maxepoch):
        tr_avg_loss, tr_avg_stage1,tr_avg_stage2,f1_stage1_avg,f1_stage2_avg = train(model1=model1,model2=model2 ,optimizer1=optimizer1,optimizer2 = optimizer2 ,dataParser = dataParser,epoch=epoch,
                                            save_dir=args.model_save_dir,)


        writer.add_scalar('tr_avg_loss_per_epoch', tr_avg_loss, global_step=epoch)
        writer.add_scalar('tr_avg_loss_stage1_per_epoch', tr_avg_stage1, global_step=epoch)
        writer.add_scalar('tr_avg_loss_stage2_per_epoch', tr_avg_stage2, global_step=epoch)
        writer.add_scalar('tr_avg_f1_stage1_per_epoch', f1_stage1_avg, global_step=epoch)
        writer.add_scalar('tr_avg_f1_stage2_per_epoch', f1_stage2_avg, global_step=epoch)
        writer.add_scalar('lr_stage2_per_epoch', scheduler2.get_lr(), global_step=epoch)
        scheduler2.step(epoch=epoch)

    print('训练已完成!')


""""""""""""""""""""""""""""""""""""
"            训练代码               "
""""""""""""""""""""""""""""""""""""


def train(model1,model2, optimizer1,optimizer2, dataParser, epoch, save_dir):
    # 读取数据的迭代器
    train_epoch = int(dataParser.steps_per_epoch)
    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    fuse_loss =Averagvalue
    f1_value_stage1 = Averagvalue()
    acc_value_stage1 = Averagvalue()
    recall_value_stage1 = Averagvalue()
    precision_value_stage1 = Averagvalue()
    map8_loss_value_stage1  =Averagvalue()
    f1_value_stage2 = Averagvalue()
    acc_value_stage2 = Averagvalue()
    recall_value_stage2 = Averagvalue()
    precision_value_stage2 = Averagvalue()
    map8_loss_value_stage2 = Averagvalue()
    loss_8 = Averagvalue()

    stage1_loss = Averagvalue()
    stage1_pred_loss = Averagvalue()
    stage2_loss = Averagvalue()
    stage2_pred_loss =Averagvalue()
    ###############################

    mid_freq = 10

    # switch to train mode
    model1.eval()
    model2.train()
    end = time.time()
    epoch_loss = []

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
            loss_stage_2 = torch.zeros(1).cuda()
            loss_8t = torch.zeros(()).cuda()
        else:
            loss = torch.zeros(1)
            loss_stage_2 = torch.zeros(1).cuda()
            loss_8t = torch.zeros(())


        with torch.set_grad_enabled(True):
            images.requires_grad = True
            # optimizer1.zero_grad()
            optimizer2.zero_grad()

            one_stage_outputs = model1(images)

            zero = torch.zeros_like(one_stage_outputs[0])
            one = torch.ones_like(one_stage_outputs[0])

            rgb_pred = images * torch.where(one_stage_outputs[0]>0.1,one,zero)
            rgb_pred_rgb = torch.cat((rgb_pred,images),1)
            _rgb_pred = rgb_pred.cpu()
            _rgb_pred = _rgb_pred.detach().numpy()
            if batch_index % mid_freq == 0:
                for i in range(args.batch_size):
                    t = _rgb_pred[i,:, :, :]
                    t = t*255
                    t = np.array(t,dtype='uint8')
                    t = t.transpose(1,2,0)
                    t = Image.fromarray(t)
                    t.save(os.path.join(two_stage_input_path,'two_stage_the_midinput_%d_%d.png'%(epoch,batch_index)))



            _rgb = images.cpu()
            _rgb = _rgb.detach().numpy()
            if batch_index % mid_freq == 0:
                for i in range(args.batch_size):
                    t = _rgb[i,:, :, :]
                    t = t*255
                    t = np.array(t,dtype='uint8')
                    t = t.transpose(1,2,0)
                    t = Image.fromarray(t)
                    t.save(os.path.join(one_stage_input_path,'two_stage_the_midinput_%d_%d.png'%(epoch,batch_index)))
            two_stage_outputs = model2(rgb_pred_rgb, one_stage_outputs[9],one_stage_outputs[10],one_stage_outputs[11])




            ##########################################
            # deal with one stage issue
            # 建立loss
            _loss_stage_1 = wce_dice_huber_loss(one_stage_outputs[0], labels[1])
            loss_stage_1 =_loss_stage_1
            ##############################################
            # deal with two stage issues
            _loss_stage_2 = wce_dice_huber_loss(two_stage_outputs[0], labels[0]) * 12

            for c_index, c in enumerate(two_stage_outputs[1:9]):
                one_loss_t = wce_dice_huber_loss(c, labels[c_index + 2])
                loss_8t += one_loss_t
                writer.add_scalar('stage2_%d_map_loss' % (c_index), one_loss_t.item(),
                                  global_step=epoch * train_epoch + batch_index)

            loss_stage_2 += loss_8t
            loss_stage_2 = _loss_stage_2 / 20


            #######################################
            # 总的LOSS
            writer.add_scalar('stage_one_loss',loss_stage_1.item(),global_step=epoch * train_epoch + batch_index)
            writer.add_scalar('stage_two_pred_loss', _loss_stage_2.item(), global_step=epoch * train_epoch + batch_index)
            writer.add_scalar('stage_two_fuse_loss', loss_stage_2.item(), global_step=epoch * train_epoch + batch_index)
            # loss = (loss_stage_1 + loss_stage_2)/2
            loss = loss_stage_2
            writer.add_scalar('fuse_loss_per_epoch', loss.item(), global_step=epoch * train_epoch + batch_index)
            ##########################################



            _output = two_stage_outputs[0].cpu()
            _output = _output.detach().numpy()
            if batch_index % mid_freq == 0:
                for i in range(args.batch_size):
                    t = _output[i, :, :]
                    t = np.squeeze(t, 0)
                    t = t*255
                    t = np.array(t,dtype='uint8')
                    t = Image.fromarray(t)

                    t.save(os.path.join(two_stage_output_path,'two_stage_the_midoutput_%d_%d.png'%(epoch,batch_index)))
            _output = one_stage_outputs[0].cpu()
            _output = _output.detach().numpy()
            if batch_index % mid_freq == 0:
                for i in range(args.batch_size):
                    t = _output[i, :, :]
                    t = np.squeeze(t, 0)
                    t = t*255
                    t = np.array(t,dtype='uint8')
                    t = Image.fromarray(t)
                    t.save(os.path.join(one_stage_output_path,'one_stage_the_midoutput_%d_%d.png'%(epoch,batch_index)))
            # 这里放保存中间结1果的代码
            # if batch_index in args.mid_result_index:
            #     save_mid_result(outputs, labels, epoch, batch_index, args.mid_result_root,save_8map=True,train_phase=True)

            loss.backward()
            # optimizer1.step()
            optimizer2.step()

        # measure the accuracy and record loss
        losses.update(loss.item())
        map8_loss_value_stage1.update(loss_8t.item())
        # epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        stage1_loss.update(loss_stage_1.item())
        stage2_loss.update(loss_stage_2.item())
        stage1_pred_loss.update(_loss_stage_1.item())
        stage2_pred_loss.update(_loss_stage_2.item())
        ##############################################
        # 评价指标
        # stage 1
        f1score_stage1 = my_f1_score(one_stage_outputs[0], labels[1])
        precisionscore_stage1 = my_precision_score(one_stage_outputs[0], labels[1])
        accscore_stage1 = my_acc_score(one_stage_outputs[0], labels[1])
        recallscore_stage1 =my_recall_score(one_stage_outputs[0],labels[1])
        # stage 2
        f1score_stage2 = my_f1_score(two_stage_outputs[0], labels[0])
        precisionscore_stage2 = my_precision_score(two_stage_outputs[0], labels[0])
        accscore_stage2 = my_acc_score(two_stage_outputs[0], labels[0])
        recallscore_stage2 = my_recall_score(two_stage_outputs[0], labels[0])
        #################################################
        writer.add_scalar('f1_score_stage1', f1score_stage1, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('precision_score_stage1', f1score_stage1, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('acc_score_stage1', f1score_stage1, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('recall_score_stage1', f1score_stage1, global_step=epoch * train_epoch + batch_index)

        writer.add_scalar('f1_score_stage2', f1score_stage2, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('precision_score_stage2', f1score_stage2, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('acc_score_stage2', f1score_stage2, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('recall_score_stage2', f1score_stage2, global_step=epoch * train_epoch + batch_index)
        ################################


        f1_value_stage1.update(f1score_stage1)
        precision_value_stage1.update(precisionscore_stage1)
        acc_value_stage1.update(accscore_stage1)
        recall_value_stage1.update(recallscore_stage1)

        f1_value_stage2.update(f1score_stage2)
        precision_value_stage2.update(precisionscore_stage2)
        acc_value_stage2.update(accscore_stage2)
        recall_value_stage2.update(recallscore_stage2)



        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.steps_per_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   '两阶段总Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   '第二阶段Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=stage2_loss) + \
                   '第一阶段:f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_stage1) + \
                   '第一阶段:precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value_stage1) + \
                   '第一阶段:acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_stage1) +\
                   '第一阶段:recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_stage1) + \
                   '第二阶段:f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_stage2) + \
                   '第二阶段:precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value_stage2) + \
                   '第二阶段:acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_stage2) +\
                   '第二阶段:recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_stage2)

            print(info)

        # 对于每一个epoch内按照一定的频率保存评价指标，以观察震荡情况
        # if batch_index % args.per_epoch_freq == 0:
        #     writer.add_scalar('tr_loss_per_epoch', losses.val, global_step=epoch * train_epoch + batch_index)
        #     writer.add_scalar('f1_score_per_epoch', f1score, global_step=epoch * train_epoch + batch_index)
        #     writer.add_scalar('precision_score_per_epoch', precisionscore,
        #                       global_step=epoch * train_epoch + batch_index)
        #     writer.add_scalar('acc_score_per_epoch', accscore, global_step=epoch * train_epoch + batch_index)
        #     writer.add_scalar('recall_score_per_epoch',recallscore,global_step=epoch * train_epoch + batch_index)
        #
        if batch_index >= train_epoch:
            break
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
        output_name = '1119checkpoint%d-stage2-%f-f1%f-precision%f-acc%f-recall%f.pth' % (epoch,losses.avg,f1_value_stage2.avg,
                                                                                                      precision_value_stage2.avg,
                                                                                                      acc_value_stage2.avg,
                                                                                                      recall_value_stage2.avg)
    try:
        send_msn(epoch,f1=f1_value_stage2.avg)
    except:
        pass




    if epoch%1 == 0:
        save_file = os.path.join(args.model_save_dir, output_name)
        # save_checkpoint({'epoch': epoch, 'state_dict': model1.state_dict(), 'optimizer': optimizer1.state_dict()},
        #                 filename=save_file)
        torch.save({'epoch': epoch, 'state_dict': model2.state_dict(), 'optimizer': optimizer2.state_dict()},save_file)

    return losses.avg, stage1_loss.avg, stage2_loss.avg, f1_value_stage1.avg,f1_value_stage2.avg


""""""""""""""""""""""""""""""""""""
"            验证代码               "
""""""""""""""""""""""""""""""""""""

#
# @torch.no_grad()
# def val(model, epoch):
#     torch.cuda.empty_cache()
#     # 读取数据的迭代器
#     dataParser = DataParser(args.batch_size)
#     #
#     batch_time = Averagvalue()
#     data_time = Averagvalue()
#     losses = Averagvalue()
#
#     # switch to train mode
#     model.eval()
#     end = time.time()
#     epoch_loss = []
#     counter = 0
#
#     for batch_index, (images, labels_numpy) in enumerate(generate_minibatches(dataParser, False)):
#
#         # 读取数据的时间
#         data_time.update(time.time() - end)
#
#         # 对读取的numpy类型数据进行调整
#         labels = []
#         if torch.cuda.is_available():
#             images = torch.from_numpy(images).cuda()
#             for item in labels_numpy:
#                 labels.append(torch.from_numpy(item).cuda())
#         else:
#             images = torch.from_numpy(images)
#             for item in labels_numpy:
#                 labels.append(torch.from_numpy(item))
#
#         # 输出结果[img，8张图]
#         outputs = model(images)
#
#         # 这里放保存中间结果的代码
#         if batch_index in args.mid_result_index:
#             save_mid_result(outputs, labels, epoch, batch_index, args.mid_result_root,save_8map=True,train_phase=False)
#
#         # 建立loss
#         loss = wce_dice_huber_loss(outputs[0], labels[0]) * 12
#         for c_index, c in enumerate(outputs[1:]):
#             loss = loss + wce_dice_huber_loss_8(c, labels[c_index + 1])
#         loss = loss / 20
#
#         # measure the accuracy and record loss
#         losses.update(loss.item(), images.size(0))
#         epoch_loss.append(loss.item())
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         # 评价指标
#         f1score = my_f1_score(outputs[0], labels[0])
#         precisionscore = my_precision_score(outputs[0], labels[0])
#         accscore = my_acc_score(outputs[0], labels[0])
#         writer.add_scalar('val_f1score', f1score, global_step=epoch * dataParser.val_steps + batch_index)
#         writer.add_scalar('val_precisionscore', precisionscore,
#                           global_step=epoch * dataParser.val_steps + batch_index)
#         writer.add_scalar('val_acc_score', accscore, global_step=epoch * dataParser.val_steps + batch_index)
#
#         if batch_index % args.print_freq == 0:
#             info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.val_steps) + \
#                    'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
#                    'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
#                    'f1_score : %.4f ' % f1score + \
#                    'precision_score: %.4f ' % precisionscore + \
#                    'acc_score %.4f ' % accscore
#
#             print('val: ',info)
#         writer.add_scalar('val_avg_loss2', losses.val, global_step=epoch * (dataParser.val_steps//100) + batch_index)
#         if batch_index > dataParser.val_steps//1:
#             break
#
#     return losses.avg, epoch_loss


if __name__ == '__main__':
    main()
