import os
import random
import logging
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from math import ceil, floor
from collections import deque
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torch.utils.data as data
from torch.autograd import Variable

import sys
sys.path.append(os.path.abspath('.'))
from utils.eval import Eval
from utils.losses import *
from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_DataLoader, GTA5_Dataset
from datasets.synthia_Dataset import SYNTHIA_Dataset

from tools.train_source import *

class STTrainer(Trainer):
    def __init__(self, args, cuda=None, train_id="None", logger=None):
        super().__init__(args, cuda, train_id, logger)
        if self.args.source_dataset == 'synthia':
            source_data_set = SYNTHIA_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split=args.source_split,
                                    base_size=args.base_size,
                                    crop_size=args.crop_size,
                                    class_16=args.class_16)
        else:
            source_data_set = GTA5_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split=args.source_split,
                                    base_size=args.base_size,
                                    crop_size=args.crop_size)
        self.source_dataloader = data.DataLoader(source_data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        if self.args.source_dataset == 'synthia':
            source_data_set = SYNTHIA_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split='val',
                                    base_size=args.base_size,
                                    crop_size=args.crop_size,
                                    class_16=args.class_16)
        else:
            source_data_set = GTA5_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split='val',
                                    base_size=args.base_size,
                                    crop_size=args.crop_size)
        self.source_val_dataloader = data.DataLoader(source_data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        print(self.args.source_dataset, self.args.target_dataset)
        target_data_set = City_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.target_base_size,
                                crop_size=args.target_crop_size,
                                class_16=args.class_16)
        self.target_dataloader = data.DataLoader(target_data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        target_data_set = City_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split='val',
                                base_size=args.target_base_size,
                                crop_size=args.target_crop_size,
                                class_16=args.class_16)
        self.target_val_dataloader = data.DataLoader(target_data_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)
        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = (len(target_data_set) + self.args.batch_size) // self.args.batch_size

        self.ignore_index = -1
        if self.args.ST_mode == "hard":
            self.target_loss = nn.CrossEntropyLoss(ignore_index= -1)
        elif self.args.ST_mode == "entropy":
            self.target_loss = softCrossEntropy(ignore_index= -1, gamma=self.args.gamma)
        elif self.args.ST_mode == "CW_entropy":
            self.target_loss = CWsoftCrossEntropy(ignore_index= -1, num_class=self.args.num_classes, gamma=self.args.gamma, threshold=self.args.soft_threshold, ratio=self.args.CW_ratio)
        elif self.args.ST_mode == "l2":
            self.target_loss = softL2loss(ignore_index= -1, num_class=self.args.num_classes, gamma=self.args.gamma, mean_prior=self.args.mean_prior)
        elif self.args.ST_mode == "CW_l2":
            self.target_loss = CWsoftL2loss(ignore_index= -1, num_class=self.args.num_classes, threshold=self.args.soft_threshold, ratio=self.args.CW_ratio, 
                                gamma=self.args.gamma, mean_prior=self.args.mean_prior)
        elif self.args.ST_mode == "scaled_entropy":
            self.target_loss = softCrossEntropy(ignore_index= -1, gamma=self.args.gamma)
        self.current_round = self.args.init_round
        self.round_num = self.args.round_num
        
        self.target_loss.to(self.device)

        if 'CW' in self.args.ST_mode:
            self.target_hard_loss = nn.CrossEntropyLoss(ignore_index= -1)
        else:
            self.target_hard_loss = nn.CrossEntropyLoss(ignore_index= -1)
        
        weight = None #torch.tensor([1,          1.3,       1,          1.1,        1,          1,           1,         1,           1,        1.1,        1,          1,         1.2,     1,          1.1,       1,         1.2,      1,         0.5]).cuda()
        self.target_label_loss = nn.CrossEntropyLoss(weight=weight, ignore_index= -1)

    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:16} {}".format(key, val))

        # choose cuda
        current_device = torch.cuda.current_device()
        self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)
        
        if not self.args.continue_training:
            self.best_MIou = 0
            self.best_iter = 0
            self.current_iter = 0
            self.current_epoch = 0

        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth'))
        
        self.args.iter_max = self.dataloader.num_iterations*self.args.epoch_each_round*self.round_num
        print(self.args.iter_max, self.dataloader.num_iterations)
        # train
        #self.validate() # check image summary
        #self.validate_source()
        self.train_round()

        self.writer.close()
    
    def train_round(self):
        for r in range(self.current_round, self.round_num):
            print("\n############## Begin {}/{} Round! #################\n".format(self.current_round+1, self.round_num))
            print("epoch_each_round:", self.args.epoch_each_round)
            
            self.epoch_num = (self.current_round+1)*self.args.epoch_each_round

            # generate threshold
            if self.args.threshold_P:
                self.cur_threshold_P = self.args.threshold_P + self.current_round*self.args.threshold_P_each_round
                self.cur_threshold_P = min(self.cur_threshold_P, 0.5)
                self.threshold = self.generate_threshold(self.cur_threshold_P)
                print("######### threshold at round {}: ({})".format(self.current_round, self.cur_threshold_P), self.threshold)
                if self.args.class_balance:
                    self.threshold_list = self.threshold
                    self.threshold = np.array(self.threshold).reshape(1, self.args.num_classes, 1, 1)
                    self.threshold = torch.from_numpy(self.threshold)
            else:
                self.threshold = self.args.threshold

            self.train()

            self.current_round += 1
        
    def train_one_epoch(self):
        tqdm_epoch = tqdm(zip(self.source_dataloader, self.target_dataloader), total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}".format(self.current_round, self.current_epoch+1, self.epoch_num))
        self.logger.info("Training one epoch...")
        self.Eval.reset()
        # Set the model to be in training mode (for batchnorm and dropout)

        train_se_loss = []
        train_target_se_loss = []
        loss_seg_value = 0
        loss_ST_value = 0
        loss_seg_value_2 = 0
        loss_ST_value_2 = 0
        iter_num = self.dataloader.num_iterations

        if self.args.freeze_bn:
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()
        # Initialize your average meters

        batch_idx = 0
        for batch_s, batch_t in tqdm_epoch:
            self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

            ##########################
            # source supervised loss #
            ##########################
            # train with source
            x, y, _ = batch_s
            if self.cuda:
                x, y = Variable(x).to(self.device), Variable(y).to(device=self.device, dtype=torch.long)

            pred = self.model(x)
            if isinstance(pred, tuple):
                pred_2 = pred[1]
                pred = pred[0]

            y = torch.squeeze(y, 1)
            loss = self.loss(pred, y)

            loss_ = loss
            if self.args.multi:
                loss_2 = self.args.lambda_seg * self.loss(pred_2, y)
                loss_ += loss_2
                loss_seg_value_2 += loss_2.cpu().item() / iter_num

            loss_.backward()
            loss_seg_value += loss.cpu().item() / iter_num
            
            ##################
            # target ST loss #
            ##################
            # train with target
            x, y, _ = batch_t
            if self.cuda:
                x, y = Variable(x).to(self.device), Variable(y).to(device=self.device, dtype=torch.long)

            pred = self.model(x)
            if isinstance(pred, tuple):
                pred_2 = pred[1]
                pred = pred[0]
                pred_P_2 = F.softmax(pred_2, dim=1)
            pred_P = F.softmax(pred, dim=1)

            if self.args.ST_mode == "hard":
                label = torch.argmax(pred_P.detach(), dim=1)
                if self.args.multi: label_2 = torch.argmax(pred_P_2.detach(), dim=1)
            else:
                label = pred_P
                if self.args.multi: label_2 = pred_P_2

            maxpred, argpred = torch.max(pred_P.detach(), dim=1)
            if self.args.multi: maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)

            if self.args.threshold_P:
                mask = (maxpred > self.threshold)
                if not self.args.class_balance and self.args.ST_mode == "hard":
                    label = torch.where(mask, label, torch.ones(1).to(self.device, dtype=torch.long)*self.ignore_index)
                    maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)
                    mask_2 = (maxpred_2 > self.threshold)
                    label_2 = torch.where(mask_2, label_2, torch.ones(1).to(self.device, dtype=torch.long)*self.ignore_index)
                elif self.args.class_balance:
                    maxpred, argpred = torch.max(pred_P.detach() /self.threshold.type_as(pred_P).to(self.device), dim=1)
                    mask = (maxpred > 1)
                    label = torch.where(mask, label, torch.ones(1).to(self.device, dtype=torch.long)*self.ignore_index)
            
            if self.args.filter:
                mask = (maxpred < self.threshold)
                label = torch.where(mask, label, torch.ones(1).to(self.device)*self.ignore_index)
            
            loss_ST = self.args.lambda_ST*self.target_loss(pred, label)

            loss_ST_ = loss_ST
            if self.args.multi:
                pred_c = (pred_P+pred_P_2)/2
                maxpred_c, argpred_c = torch.max(pred_c, dim=1)
                mask = (maxpred > self.threshold) | (maxpred_2 > self.threshold)

                label_2 = torch.where(mask, argpred_c, torch.ones(1).to(self.device, dtype=torch.long)*self.ignore_index)
                loss_ST_2 = self.args.lambda_seg * self.args.lambda_ST*self.target_hard_loss(pred_2, label_2)
                loss_ST_ += loss_ST_2
                loss_ST_value_2 += loss_ST_2 / iter_num
            
            if self.args.target_label:
                loss_ST_ += self.args.lambda_ST*self.target_label_loss(pred, y)
                if self.args.multi:
                    loss_ST_ += self.args.lambda_seg * self.args.lambda_ST * self.target_label_loss(pred_2, y)

            loss_ST_.backward()
            loss_ST_value += loss_ST / iter_num

            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch_idx % 400 == 0:
                if self.args.multi:
                    self.logger.info("epoch{}-batch-{}:loss_seg={:.3f}-loss_ST={:.3f}; loss_seg_2={:.3f}-loss_ST_2={:.3f}".format(self.current_epoch,
                                                                           batch_idx, loss.item(), loss_ST.item(), loss_2.item(), loss_ST_2.item()))
                else:
                    self.logger.info("epoch{}-batch-{}:loss_seg={:.3f}-loss_ST={:.3f}".format(self.current_epoch,
                                                                           batch_idx, loss.item(), loss_ST.item()))
            batch_idx += 1

            self.current_iter += 1

            pred = pred.data.cpu().numpy()
            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            self.Eval.add_batch(label, argpred)
        
        self.log_one_train_epoch(x, label, argpred, loss_seg_value)
        self.writer.add_scalar('ST_loss', loss_ST_value, self.current_epoch)
        tqdm.write("The average ST_loss of train epoch-{}-:{:.3f}".format(self.current_epoch, loss_ST_value))
        if self.args.multi:
            self.writer.add_scalar('train_loss_2', loss_seg_value_2, self.current_epoch)
            tqdm.write("The average loss_2 of train epoch-{}-:{}".format(self.current_epoch, loss_seg_value_2))
            self.writer.add_scalar('ST_loss_2', loss_ST_value_2, self.current_epoch)
            tqdm.write("The average ST_loss_2 of train epoch-{}-:{:.3f}".format(self.current_epoch, loss_ST_value_2))
        tqdm_epoch.close()
        
        #eval on source domain
        self.validate_source()

    def generate_threshold(self, threshold_P, pixel_cls=None):
        if pixel_cls is None:
            target_data_set = City_Dataset(self.args,
                split='train',
                base_size=self.args.target_base_size,
                crop_size=self.args.target_crop_size,
                training=False)
            target_dataloader = data.DataLoader(target_data_set,
                                                batch_size=self.args.batch_size,
                                                shuffle=False,
                                                num_workers=self.args.data_loader_workers,
                                                pin_memory=self.args.pin_memory,
                                                drop_last=True)
            max_i = 500 if self.args.class_balance else 100
            tqdm_epoch = tqdm(target_dataloader, total=min(len(target_dataloader), max_i),
                            desc="generate_threshold")
            self.logger.info("Generating threshold...")

            pixel_cls = []
            i = 0
            for x, _y, _ in tqdm_epoch:
                if self.cuda:
                    x = x.to(self.device)
                pred = self.model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                if self.args.binary_loss:
                    pred_P = robust_sigmoid(pred, temperature=self.args.temperature)
                else:
                    pred_P = robust_softmax(pred, dim=1, temperature=self.args.temperature)
                pred_P = pred_P.data.cpu().numpy()
                if self.args.spatial_prior:
                    pred_P = pred_P*self.spatial_prior[None,:]
                pred_P = pred_P.transpose(0,2,3,1).reshape(-1,self.args.num_classes)
                pred_P = np.uint16(pred_P*1000)
                pixel_cls.append(pred_P)
                i += 1
                if i==max_i:
                    break
            tqdm_epoch.close()

        array_cls = np.concatenate(pixel_cls, axis=0)
        if pixel_cls is None: print(array_cls.shape)

        # threshold for each class
        if not self.args.class_balance:
            array_pixel = np.max(array_cls, axis=1)
            array_pixel = sorted(array_pixel, reverse = True)
            len_cls = len(array_pixel)
            len_thresh = int(floor(len_cls * threshold_P))
            cls_thresh = min(max(array_pixel[len_thresh-1].copy() / 1000, 0.0), 0.98)
            len_thresh_2 = int(floor(len_cls * 0.7))
            print("threshold:", array_pixel[len_thresh-1] / 1000, array_pixel[int(floor(len_cls * 0.4))-1] / 1000, array_pixel[len_thresh_2-1] / 1000)
            return cls_thresh
        else:
            from datasets.cityscapes_Dataset import name_classes
            cls_thresh_CW = []
            max_idx = np.argmax(array_cls, axis=1)
            for idx_cls in range(self.args.num_classes):
                array_pixel = array_cls[:,idx_cls][max_idx==idx_cls]
                len_cls = len(array_pixel)
                if len_cls > 0:
                    array_pixel = sorted(array_pixel, reverse = True)
                    len_thresh = int(floor(len_cls * threshold_P))
                    thresh = min(max(array_pixel[len_thresh-1].copy() / 1000, 0.0), 0.98)
                    cls_thresh_CW.append(thresh)
                    if pixel_cls is None: print(name_classes[idx_cls], len_cls, thresh)
                else:
                    cls_thresh_CW.append(1.0)
            return cls_thresh_CW

def add_ST_train_args(arg_parser):
    arg_parser.add_argument('--source_dataset', default='gta5', type=str,
                            help='source dataset choice')
    arg_parser.add_argument('--source_split', default='train', type=str,
                            help='source_split')
    arg_parser.add_argument('--round_num', type=int, default=1,
                            help="num round")
    arg_parser.add_argument('--epoch_each_round', type=int, default=2,
                            help="epoch_each_round")
    arg_parser.add_argument('--threshold_P_each_round', type=float, default=0.05,
                            help="threshold_P grows each_round")                        
    arg_parser.add_argument('--ST_mode', type=str, default="hard",
                            help="ST_mode")
    arg_parser.add_argument('--class_balance', type=str2bool, default=False,
                            help="whether to use class_balance for pseudo-labels")
    arg_parser.add_argument('--threshold_P', type=float, default=0,
                            help="use px100% most confident pseudo-labels")
    arg_parser.add_argument('--soft_threshold', type=float, default=0.0, 
                            help='soft_threshold')
    arg_parser.add_argument('--lambda_ST', type=float, default=1,
                            help="lambda_ST")
    arg_parser.add_argument('--with_prior', type=str2bool, default=False,
                            help='with prior')
    arg_parser.add_argument('--gamma', type=float, default=0, 
                            help='stable entorpy')
    arg_parser.add_argument('--CW_ratio', type=float, default=0.2, 
                            help='CW_ratio')
    arg_parser.add_argument('--init_round', type=int, default=0, 
                            help='init_round')
    arg_parser.add_argument('--spatial_prior', type=str2bool, default=False,
                            help='spatial_prior')
    arg_parser.add_argument('--mean_prior', type=str2bool, default=False,
                            help='mean_prior')
    arg_parser.add_argument('--all_gta5', type=str2bool, default=False,
                            help='all_gta5')
    arg_parser.add_argument('--target_label', type=str2bool, default=False,
                            help='target_label')
    arg_parser.add_argument('--filter', type=str2bool, default=False,
                            help='filter')
    arg_parser.add_argument('--threshold', type=float, default=0.98,
                            help="threshold")
    return arg_parser

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_ST_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)
    args.source_data_path = datasets_path[args.source_dataset]['data_root_path']
    args.source_list_path = datasets_path[args.source_dataset]['list_path']

    args.target_dataset = args.dataset

    agent = STTrainer(args=args, cuda=True, train_id="train_id", logger=logger)
    agent.main()