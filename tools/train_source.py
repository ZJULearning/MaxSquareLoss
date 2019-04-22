import os
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
from math import ceil
import numpy as np
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

import sys
sys.path.append(os.path.abspath('.'))
from utils.eval import Eval
from utils.train_helper import get_model

from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_DataLoader
from datasets.synthia_Dataset import SYNTHIA_DataLoader


datasets_path={
    'cityscapes': {'data_root_path': './datasets/Cityscapes', 'list_path': './datasets/city_list', 
                    'image_path':'./datasets/Cityscapes/leftImg8bit',
                    'gt_path': './datasets/Cityscapes/gtFine'},
    'gta5': {'data_root_path': './datasets/GTA5', 'list_path': './datasets/GTA5/list',
                    'image_path':'./datasets/GTA5/images',
                    'gt_path': './datasets/GTA5/labels'},
    'synthia': {'data_root_path': './datasets/SYNTHIA', 'list_path': './datasets/SYNTHIA/list',
                    'image_path':'./datasets/SYNTHIA/RGB',
                    'gt_path': './datasets/SYNTHIA/GT/LABELS'},
    'NTHU': {'data_root_path': './datasets/NTHU_Datasets', 'list_path': './datasets/NTHU_list'}
    }

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

ITER_MAX = 5000

class Trainer():
    def __init__(self, args, cuda=None, train_id="None", logger=None):
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.train_id = train_id
        self.logger = logger

        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.second_best_MIou = 0

        # set TensorboardX
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        self.loss = nn.CrossEntropyLoss(weight=None, ignore_index= -1)
        self.loss.to(self.device)

        # model
        self.model, params = get_model(self.args)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model.to(self.device)

        if self.args.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                params=params,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optim == "Adam":
            self.optimizer = torch.optim.Adam(params, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        # dataloader
        if self.args.dataset=="cityscapes":
            self.dataloader = City_DataLoader(self.args)  
        elif self.args.dataset=="gta5":
            self.dataloader = GTA5_DataLoader(self.args)
        else:
            self.dataloader = SYNTHIA_DataLoader(self.args)
        self.dataloader.num_iterations = min(self.dataloader.num_iterations, ITER_MAX)
        print(self.args.iter_max, self.dataloader.num_iterations)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.num_iterations) if self.args.iter_stop is None else \
                            ceil(self.args.iter_stop / self.dataloader.num_iterations)

    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:16} {}".format(key, val))

        # choose cuda
        if self.cuda:
            current_device = torch.cuda.current_device()
            self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            self.logger.info("This model will run on CPU")

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'best.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)
        
        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.train_id + 'best.pth'))
            self.best_iter = self.current_iter
            self.best_source_iter = self.current_iter
        else:
            self.current_epoch = 0
        # train
        self.train()

        self.writer.close()

    def train(self):
        # self.validate() # check image summary

        for epoch in tqdm(range(self.current_epoch, self.epoch_num),
                          desc="Total {} epochs".format(self.epoch_num)):
            self.train_one_epoch()

            # validate
            PA, MPA, MIoU, FWIoU = self.validate()
            self.writer.add_scalar('PA', PA, self.current_epoch)
            self.writer.add_scalar('MPA', MPA, self.current_epoch)
            self.writer.add_scalar('MIoU', MIoU, self.current_epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.current_epoch)

            self.current_MIoU = MIoU
            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
                self.best_iter = self.current_iter
                self.logger.info("=>saving a new best checkpoint...")
                self.save_checkpoint(self.train_id+'best.pth')
            else:
                self.logger.info("=> The MIoU of val does't improve.")
                self.logger.info("=> The best MIoU of val is {} at {}".format(self.best_MIou, self.best_iter))
            
            self.current_epoch += 1

        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.current_MIoU
        }
        self.logger.info("=>best_MIou {} at {}".format(self.best_MIou, self.best_iter))
        self.logger.info("=>saving the final checkpoint to " + os.path.join(self.args.checkpoint_dir, self.train_id+'final.pth'))
        self.save_checkpoint(self.train_id+'final.pth')

    def train_one_epoch(self):
        tqdm_epoch = tqdm(self.dataloader.data_loader, total=self.dataloader.num_iterations,
                          desc="Train Epoch-{}-total-{}".format(self.current_epoch+1, self.epoch_num))
        self.logger.info("Training one epoch...")
        self.Eval.reset()

        train_loss = []
        loss_seg_value_2 = 0
        iter_num = self.dataloader.num_iterations
        
        if self.args.freeze_bn:
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()
        # Initialize your average meters

        batch_idx = 0
        for x, y, _ in tqdm_epoch:
            self.poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=self.args.lr,
                iter=self.current_iter,
                max_iter=self.args.iter_max,
                power=self.args.poly_power,
            )
            if self.args.iter_stop is not None and self.current_iter >= self.args.iter_stop:
                self.logger.info("iteration arrive {}(early stop)/{}(total step)!".format(self.args.iter_stop, self.args.iter_max))
                break
            if self.current_iter >= self.args.iter_max:
                self.logger.info("iteration arrive {}!".format(self.args.iter_max))
                break
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)
            y = torch.squeeze(y, 1)
            self.optimizer.zero_grad()

            # model
            pred = self.model(x)
            if isinstance(pred, tuple):
                pred_2 = pred[1]
                pred = pred[0]
            
            # loss
            cur_loss = self.loss(pred, y)
            
            if self.args.multi:
                loss_2 = self.args.lambda_seg * self.loss(pred_2, y)
                cur_loss += loss_2
                loss_seg_value_2 += loss_2.cpu().item() / iter_num

            # optimizer
            cur_loss.backward()
            self.optimizer.step()

            train_loss.append(cur_loss.item())

            if batch_idx % 1000 == 0:
                if self.args.multi:
                    self.logger.info("The train loss of epoch{}-batch-{}:{};{}".format(self.current_epoch,
                                                                            batch_idx, cur_loss.item(), loss_2.item()))
                else:
                    self.logger.info("The train loss of epoch{}-batch-{}:{}".format(self.current_epoch,
                                                                            batch_idx, cur_loss.item()))
                
            batch_idx += 1

            self.current_iter += 1

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            pred = pred.data.cpu().numpy()
            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            self.Eval.add_batch(label, argpred)

            if batch_idx==self.dataloader.num_iterations:
                break
        
        self.log_one_train_epoch(x, label, argpred, train_loss)
        tqdm_epoch.close()

    def log_one_train_epoch(self, x, label, argpred, train_loss):
        #show train image on tensorboard
        images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
        labels_colors = decode_labels(label, self.args.show_num_images)
        preds_colors = decode_labels(argpred, self.args.show_num_images)
        for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image('train/'+ str(index)+'/Images', img, self.current_epoch)
            self.writer.add_image('train/'+ str(index)+'/Labels', lab, self.current_epoch)
            self.writer.add_image('train/'+ str(index)+'/preds', color_pred, self.current_epoch)

        if self.args.class_16:
            PA = self.Eval.Pixel_Accuracy()
            MPA_16, MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU_16, MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()
        else:
            PA = self.Eval.Pixel_Accuracy()
            MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()

        self.logger.info('\nEpoch:{}, train PA1:{}, MPA1:{}, MIoU1:{}, FWIoU1:{}'.format(self.current_epoch, PA, MPA,
                                                                                       MIoU, FWIoU))
        self.writer.add_scalar('train_PA', PA, self.current_epoch)
        self.writer.add_scalar('train_MPA', MPA, self.current_epoch)
        self.writer.add_scalar('train_MIoU', MIoU, self.current_epoch)
        self.writer.add_scalar('train_FWIoU', FWIoU, self.current_epoch)

        tr_loss = sum(train_loss)/len(train_loss) if isinstance(train_loss, list) else train_loss
        self.writer.add_scalar('train_loss', tr_loss, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, tr_loss))

    def validate(self, mode='val'):
        self.logger.info('\nvalidating one epoch...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.val_loader, total=self.dataloader.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch + 1))
            if mode == 'val':
                self.model.eval()
            
            i = 0

            for x, y, id in tqdm_batch:
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)                   
                if isinstance(pred, tuple):
                    pred_2 = pred[1]
                    pred = pred[0]
                    pred_P = F.softmax(pred, dim=1)
                    pred_P_2 = F.softmax(pred_2, dim=1)
                y = torch.squeeze(y, 1)


                pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)
                

            #show val result on tensorboard
            images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
            labels_colors = decode_labels(label, self.args.show_num_images)
            preds_colors = decode_labels(argpred, self.args.show_num_images)
            for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                self.writer.add_image(str(index)+'/Images', img, self.current_epoch)
                self.writer.add_image(str(index)+'/Labels', lab, self.current_epoch)
                self.writer.add_image(str(index)+'/preds', color_pred, self.current_epoch)

            if self.args.class_16:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16,
                                                                                                MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13,
                                                                                                MIoU_13, FWIoU_13, PC_13))
                    self.writer.add_scalar('PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA_16'+name, MPA_16, self.current_epoch)
                    self.writer.add_scalar('MIoU_16'+name, MIoU_16, self.current_epoch)
                    self.writer.add_scalar('FWIoU_16'+name, FWIoU_16, self.current_epoch)
                    self.writer.add_scalar('MPA_13'+name, MPA_13, self.current_epoch)
                    self.writer.add_scalar('MIoU_13'+name, MIoU_13, self.current_epoch)
                    self.writer.add_scalar('FWIoU_13'+name, FWIoU_13, self.current_epoch)
                    return PA, MPA_13, MIoU_13, FWIoU_13
            else:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA,
                                                                                                MIoU, FWIoU, PC))
                    self.writer.add_scalar('PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA'+name, MPA, self.current_epoch)
                    self.writer.add_scalar('MIoU'+name, MIoU, self.current_epoch)
                    self.writer.add_scalar('FWIoU'+name, FWIoU, self.current_epoch)
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU

    def validate_source(self):
        self.logger.info('\nvalidating source domain...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.source_val_dataloader, total=self.dataloader.valid_iterations,
                              desc="Source Val Epoch-{}-".format(self.current_epoch + 1))
            self.model.eval()
            i = 0
            for x, y, id in tqdm_batch:
                # y.to(torch.long)
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)

                if isinstance(pred, tuple):
                    pred_2 = pred[1]
                    pred = pred[0]
                    pred_P = F.softmax(pred, dim=1)
                    pred_P_2 = F.softmax(pred_2, dim=1)
                y = torch.squeeze(y, 1)

                pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)

                i += 1
                if i == self.dataloader.valid_iterations:
                    break

            #show val result on tensorboard
            images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
            labels_colors = decode_labels(label, self.args.show_num_images)
            preds_colors = decode_labels(argpred, self.args.show_num_images)
            for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                self.writer.add_image('source_eval/'+str(index)+'/Images', img, self.current_epoch)
                self.writer.add_image('source_eval/'+str(index)+'/Labels', lab, self.current_epoch)
                self.writer.add_image('source_eval/'+str(index)+'/preds', color_pred, self.current_epoch)

            if self.args.class_16:
                def source_val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Source Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, source {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16,
                                                                                                MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, source {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13,
                                                                                                MIoU_13, FWIoU_13, PC_13))
                    self.writer.add_scalar('source_PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('source_MPA_16'+name, MPA_16, self.current_epoch)
                    self.writer.add_scalar('source_MIoU_16'+name, MIoU_16, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU_16'+name, FWIoU_16, self.current_epoch)
                    self.writer.add_scalar('source_MPA_13'+name, MPA_13, self.current_epoch)
                    self.writer.add_scalar('source_MIoU_13'+name, MIoU_13, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU_13'+name, FWIoU_13, self.current_epoch)
                    return PA, MPA_13, MIoU_13, FWIoU_13
            else:
                def source_val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()

                    self.writer.add_scalar('source_PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('source_MPA'+name, MPA, self.current_epoch)
                    self.writer.add_scalar('source_MIoU'+name, MIoU, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU'+name, FWIoU, self.current_epoch)
                    print("########## Source Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, source {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA,
                                                                                                MIoU, FWIoU, PC))
                    return PA, MPA, MIoU, FWIoU
        
            PA, MPA, MIoU, FWIoU = source_val_info(self.Eval, "")
            tqdm_batch.close()

        is_best = MIoU > self.best_source_MIou
        if is_best:
            self.best_source_MIou = MIoU
            self.best_source_iter = self.current_iter
            self.logger.info("=>saving a new best source checkpoint...")
            self.save_checkpoint(self.train_id+'source_best.pth')
        else:
            self.logger.info("=> The source MIoU of val does't improve.")
            self.logger.info("=> The best source MIoU of val is {} at {}".format(self.best_source_MIou, self.best_source_iter))

        return PA, MPA, MIoU, FWIoU

    def save_checkpoint(self, filename=None):
        """
        Save checkpoint if a new best is achieved
        :param state:
        :param is_best:
        :param filepath:
        :return:
        """
        filename = os.path.join(self.args.checkpoint_dir, filename)
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou':self.best_MIou
        }
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from "+filename)
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            self.logger.info("**First time to train**")

    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None, 
                            max_iter=None, power=None):
        init_lr = self.args.lr if init_lr is None else init_lr
        iter = self.current_iter if iter is None else iter
        max_iter = self.args.iter_max if max_iter is None else max_iter
        power = self.args.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[1]["lr"] = 10 * new_lr


def add_train_args(arg_parser):
    # Path related arguments
    arg_parser.add_argument('--data_root_path', type=str, default=None,
                            help="the root path of dataset")
    arg_parser.add_argument('--list_path', type=str, default=None,
                            help="the root path of dataset")
    arg_parser.add_argument('--checkpoint_dir', default="./log/train",
                            help="the path of ckpt file")

    # Model related arguments
    arg_parser.add_argument('--backbone', default='deeplabv2_multi',
                            help="backbone of encoder")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1,
                            help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply imagenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=None,
                            help="whether apply pretrained checkpoint")
    arg_parser.add_argument('--continue_training', type=str2bool, default=False,
                            help="whether to continue training ")
    arg_parser.add_argument('--show_num_images', type=int, default=2,
                        help="show how many images during validate")

    # train related arguments
    arg_parser.add_argument('--seed', default=12345, type=int,
                            help='random seed')
    arg_parser.add_argument('--gpu', type=str, default="0",
                            help=" the num of gpu")
    arg_parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                            help='input batch size')

    # dataset related arguments
    arg_parser.add_argument('--dataset', default='cityscapes', type=str,
                            help='dataset choice')
    arg_parser.add_argument('--base_size', default="1280,720", type=str,
                            help='crop size of image')
    arg_parser.add_argument('--crop_size', default="1280,720", type=str,
                            help='base size of image')
    arg_parser.add_argument('--target_base_size', default="1024,512", type=str,
                            help='crop size of target image')
    arg_parser.add_argument('--target_crop_size', default="1024,512", type=str,
                            help='base size of target image')
    arg_parser.add_argument('--num_classes', default=19, type=int,
                            help='num class of mask')
    arg_parser.add_argument('--data_loader_workers', default=16, type=int,
                            help='num_workers of Dataloader')
    arg_parser.add_argument('--pin_memory', default=2, type=int,
                            help='pin_memory of Dataloader')
    arg_parser.add_argument('--split', type=str, default='train',
                            help="choose from train/val/test/trainval/all")
    arg_parser.add_argument('--random_mirror', default=True, type=str2bool,
                            help='add random_mirror')
    arg_parser.add_argument('--random_crop', default=False, type=str2bool,
                        help='add random_crop')
    arg_parser.add_argument('--resize', default=True, type=str2bool,
                        help='resize')
    arg_parser.add_argument('--gaussian_blur', default=True, type=str2bool,
                        help='add gaussian_blur')
    arg_parser.add_argument('--numpy_transform', default=True, type=str2bool,
                        help='image transform with numpy style')

    # optimization related arguments

    arg_parser.add_argument('--freeze_bn', type=str2bool, default=False,
                            help="whether freeze BatchNormalization")
    arg_parser.add_argument('--optim', default="SGD", type=str,
                            help='optimizer')
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4)

    arg_parser.add_argument('--lr', type=float, default=2.5e-4,
                            help="init learning rate ")
    arg_parser.add_argument('--iter_max', type=int, default=250000,
                            help="the maxinum of iteration")
    arg_parser.add_argument('--iter_stop', type=int, default=None,
                            help="the early stop step")
    arg_parser.add_argument('--poly_power', type=float, default=0.9,
                            help="poly_power")

    # multi-level output

    arg_parser.add_argument('--multi', default=False, type=str2bool,
                        help='output model middle feature')
    arg_parser.add_argument('--lambda_seg', type=float, default=0.1,
                        help="lambda_seg of middle output")
    return arg_parser

def init_args(args):
    args.batch_size = args.batch_size_per_gpu * ceil(len(args.gpu) / 2)
    print("batch size: ", args.batch_size)

    train_id = str(args.dataset)

    crop_size = args.crop_size.split(',')
    base_size = args.base_size.split(',')
    if len(crop_size)==1:
        args.crop_size = int(crop_size[0])
        args.base_size = int(base_size[0])
    else:
        args.crop_size = (int(crop_size[0]), int(crop_size[1]))
        args.base_size = (int(base_size[0]), int(base_size[1]))

    target_crop_size = args.target_crop_size.split(',')
    target_base_size = args.target_base_size.split(',')
    if len(target_crop_size)==1:
        args.target_crop_size = int(target_crop_size[0])
        args.target_base_size = int(target_base_size[0])
    else:
        args.target_crop_size = (int(target_crop_size[0]), int(target_crop_size[1]))
        args.target_base_size = (int(target_base_size[0]), int(target_base_size[1]))

    if not args.continue_training:
        if os.path.exists(args.checkpoint_dir):
            print("checkpoint dir exists, which will be removed")
            import shutil
            shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
        os.mkdir(args.checkpoint_dir)

    if args.data_root_path is None:
        args.data_root_path = datasets_path[args.dataset]['data_root_path']
        args.list_path = datasets_path[args.dataset]['list_path']
        args.image_filepath = datasets_path[args.dataset]['image_path']
        args.gt_filepath = datasets_path[args.dataset]['gt_path']
    
    args.class_16 = True if args.num_classes == 16 else False
    args.class_13 = True if args.num_classes == 13 else False

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.checkpoint_dir, 'train_log.txt'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=True

    return args, train_id, logger
    
if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)

    agent = Trainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()