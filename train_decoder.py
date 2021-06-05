from model import Decoder
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

import numpy as np

from torchvision import transforms, datasets
import torchvision

from util import *
from dataset import *

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print every print_freq batchs')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save model checkpoint every save_freq epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product') # dimension of network's output
    parser.add_argument('--img_size', type=int, default=224) # 生成的图像尺寸
    parser.add_argument('--feat_path', type=str, default='') # 编码器计算得到的训练数据的特征文件

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to training data') # 训练数据文件夹，即锚点/正负样本文件夹
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')
    parser.add_argument('--log_txt_path', type=str, default=None, help='path to log file')
    parser.add_argument('--result_path', type=str, default=None, help='path to sample dis and img case study') # 训练结束后，对比生成的图像和原始图像，保存到这个文件夹下


    parser.add_argument('--comment_info', type=str, default='', help='Comment message, donot influence program')

    parser.add_argument('--training_data_cache_method', type=str, default='default', choices=['default', 'memory', 'GPU'], help='where to save training data. \'memory\' or \'GPU\' will load all training data into memory or GPU at begining to speed up data reading at training stage.')

    opt = parser.parse_args()

    curTime = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())

    opt.model_name = '{}_lr_{}_bsz_{}_featDim_{}_{}'.format(curTime, opt.lr, opt.batch_size, opt.feat_dim, opt.comment_info)
    
    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None) or (opt.log_txt_path is None) or (opt.feat_path is None) or (opt.result_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path | log_txt_path | feat_path | result_path')

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.log_txt_path):
        os.makedirs(opt.log_txt_path)

    opt.result_path = os.path.join(opt.result_path, opt.model_name)
    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)
    
    log_file_name = os.path.join(opt.log_txt_path, 'log_'+opt.model_name+'.txt') 
    sys.stdout = Logger(log_file_name) # 把print的东西输出到txt文件中

    print('==============================Train Decoder==============================')
    if opt.comment_info != '':
        print('comment message : ', opt.comment_info)

    print('start program at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    print('imgPath :', opt.data_folder)
    print('featPath :', opt.feat_path)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt

def get_train_loader(args):
    data_folder = os.path.join(args.data_folder, 'train')

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 选择训练数据读取方式
    # default：传统方式，即每次仅读取一个batch的数据到内存中；
    # memory 或 GPU：在程序开始阶段就将所有数据读到内存或显存中，以加快训练过程中的数据读取速度，但面临内存或显存不够用的问题
    if args.training_data_cache_method == 'default':
        train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    else:
        if torch.cuda.is_available() and args.training_data_cache_method == 'GPU':
            train_dataset = ImageFolderInstance_LoadAllImgToMemory(data_folder, transform=train_transform, training_data_cache_method='GPU')
        else:
            train_dataset = ImageFolderInstance_LoadAllImgToMemory(data_folder, transform=train_transform, training_data_cache_method='memory')
            if (not torch.cuda.is_available()) and args.training_data_cache_method == 'GPU':
                print('CUDA is not is_available, load all training data into memory instead of GPU')

    pin_memory = True
    if args.training_data_cache_method == 'GPU' and torch.cuda.is_available():
        pin_memory = False # 如果所有训练数据已经放在显存里了，那就不能再设置这个选项了。这个选项的含义是锁页内存，锁页内存中的数据永远不会与虚拟内存交换，即放在这里面的数据可以有很高的IO

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=pin_memory)

    feat = np.load(args.feat_path)
    feat = torch.from_numpy(feat)

    return train_loader, feat, train_dataset


def set_model(args):
    model = Decoder(feat_dim=args.feat_dim, img_size=args.img_size)

    if args.resume:
        if torch.cuda.is_available():
            ckpt = torch.load(args.resume)
        else:
            ckpt = torch.load(args.resume,map_location=torch.device('cpu'))
        print("==> loaded pre-trained checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))
        model.load_state_dict(ckpt['model'])
        print('==> done')
    
    criterion = torch.nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    
    return model, criterion

def set_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    return optimizer

def train_one_epoch(epoch, train_loader, feat, model, criterion, optimizer, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx,(img, target, index) in enumerate(train_loader):
        cur_feat = feat[index]
        data_time.update(time.time() - end)

        bsz = img.size(0)
        if torch.cuda.is_available():
            img = img.cuda()
            cur_feat = cur_feat.cuda()

        # ===================forward=====================
        generated_img = model(cur_feat)
        loss = criterion(generated_img, img)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

                # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg



def main():
    args = parse_option()
    args.start_epoch = 1
    train_loader, feat, train_dataset = get_train_loader(args)
    model, criterion = set_model(args)
    optimizer = set_optimizer(args, model)

    print('start training at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    min_loss = np.inf
    best_model_path = ''

    for epoch in range(args.start_epoch, args.epochs + 1):

        loss = train_one_epoch(epoch, train_loader, feat, model, criterion, optimizer, args)

        if epoch % 5 == 0:
            imageCaseStudy(train_dataset,feat,model,args, epoch)

        print_running_time(start_time)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        if loss < min_loss:
            if min_loss != np.inf:
                os.remove(best_model_path)
            min_loss = loss
            best_model_path = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}_Best.pth'.format(epoch=epoch))
            print('==> Saving best model...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, best_model_path)
            # help release GPU memory
            del state


if __name__ == '__main__':
    main()