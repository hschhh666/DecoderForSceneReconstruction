from __future__ import print_function
from numpy.core.defchararray import mod

import torch
import numpy as np
import time
import sys
import os
import cv2
import random

def print_running_time(start_time):
    print()
    print('='*20,end = ' ')
    print('Time is ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,end = ' ')
    using_time = time.time()-start_time
    hours = int(using_time/3600)
    using_time -= hours*3600
    minutes = int(using_time/60)
    using_time -= minutes*60
    print('running %d h,%d m,%d s'%(hours,minutes,int(using_time)),end = ' ')
    print('='*20)
    print()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object): # 旨在把程序中所有print出来的内容都保存到文件中
    def __init__(self, filename="Default.log"):
        path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(path,filename)
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass

def imageCaseStudy(dataset, feat, model, args, name):
    
    def imgConvent(img):
        img = img[0] # 别问为啥，问就是试出来的
        mean=[0.485, 0.456, 0.406] 
        std=[0.229, 0.224, 0.225]
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        for i in range(3):
            img[:,:,i] = img[:,:,i]*std[i]+mean[i] # 把原本normalize的图像变回来
        tmp = img.copy()
        img[:,:,0] = tmp[:,:,2]
        img[:,:,2] = tmp[:,:,0]
        return img

    img_size = args.img_size
    horizon_intervel = 10
    vertical_intervel = 5
    res_img = []

    n_data = len(dataset)
    sample_idxes = random.sample(range(n_data), 8) # 采样八个
    for i, idx in enumerate(sample_idxes):
        img = dataset[idx][0].unsqueeze(dim=0)
        cur_feat = feat[idx].unsqueeze(dim=0)
        if torch.cuda.is_available():
            img = img.cuda()
            cur_feat = cur_feat.cuda()
        generated_img = model(cur_feat)
        img = img.detach().cpu()
        generated_img = generated_img.detach().cpu()
        img = imgConvent(img)
        generated_img = imgConvent(generated_img)
        cur_img = np.concatenate((img, generated_img), axis=1)
        if res_img == []:
            res_img = cur_img
        else:
            res_img = np.concatenate((res_img, cur_img), axis=1)
    res_img[res_img > 1] = 1
    res_img = res_img * 255
    res_img = res_img.astype(np.uint8)    
    tmp = res_img[:,0*img_size*4:(0+1)*img_size*4,:]
    for i in range(1,4):
        tmp = np.concatenate((tmp, res_img[:,i*img_size*4:(i+1)*img_size*4,:]), axis=0)
    res_img = tmp
    img_name = os.path.join(args.result_path,str(name) + '.png')
    cv2.imwrite(img_name, res_img)




if __name__ == '__main__':
    meter = AverageMeter()
