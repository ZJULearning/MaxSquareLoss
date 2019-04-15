# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 13:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : eval.py
# @Software: PyCharm

import os
import time
import numpy as np
import torch
from PIL import Image
from datasets.cityscapes_Dataset import name_classes

np.seterr(divide='ignore', invalid='ignore')

synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_16_to_13 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

class Eval():
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.ignore_index = None
        self.synthia = True if num_class == 16 else False


    def Pixel_Accuracy(self):
        if np.sum(self.confusion_matrix) == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return PA

    def Mean_Pixel_Accuracy(self, out_16_13=False):
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.synthia:
            MPA_16 = np.nanmean(MPA[:self.ignore_index])
            MPA_13 = np.nanmean(MPA[synthia_set_16_to_13])
            return MPA_16, MPA_13
        if out_16_13:
            MPA_16 = np.nanmean(MPA[synthia_set_16])
            MPA_13 = np.nanmean(MPA[synthia_set_13])
            return MPA_16, MPA_13
        MPA = np.nanmean(MPA[:self.ignore_index])

        return MPA

    def Mean_Intersection_over_Union(self, out_16_13=False):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        if self.synthia:
            MIoU_16 = np.nanmean(MIoU[:self.ignore_index])
            MIoU_13 = np.nanmean(MIoU[synthia_set_16_to_13])
            return MIoU_16, MIoU_13
        if out_16_13:
            MIoU_16 = np.nanmean(MIoU[synthia_set_16])
            MIoU_13 = np.nanmean(MIoU[synthia_set_13])
            return MIoU_16, MIoU_13
        MIoU = np.nanmean(MIoU[:self.ignore_index])

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self, out_16_13=False):
        FWIoU = np.multiply(np.sum(self.confusion_matrix, axis=1), np.diag(self.confusion_matrix))
        FWIoU = FWIoU / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                         np.diag(self.confusion_matrix))
        if self.synthia:
            FWIoU_16 = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)
            FWIoU_13 = np.sum(i for i in FWIoU[synthia_set_16_to_13] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        if out_16_13:
            FWIoU_16 = np.sum(i for i in FWIoU[synthia_set_16] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            FWIoU_13 = np.sum(i for i in FWIoU[synthia_set_13] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        FWIoU = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)

        return FWIoU

    def Mean_Precision(self, out_16_13=False):
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        if self.synthia:
            Precision_16 = np.nanmean(Precision[:self.ignore_index])
            Precision_13 = np.nanmean(Precision[synthia_set_16_to_13])
            return Precision_16, Precision_13
        if out_16_13:
            Precision_16 = np.nanmean(Precision[synthia_set_16])
            Precision_13 = np.nanmean(Precision[synthia_set_13])
            return Precision_16, Precision_13
        Precision = np.nanmean(Precision[:self.ignore_index])
        return Precision
    
    def Print_Every_class_Eval(self, out_16_13=False):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Class_ratio = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        Pred_retio = np.sum(self.confusion_matrix, axis=0) / np.sum(self.confusion_matrix)
        print('===>Everyclass:\t' + 'MPA\t' + 'MIoU\t' + 'PC\t' + 'Ratio\t' + 'Pred_Retio')
        if out_16_13: MIoU = MIoU[synthia_set_16]
        for ind_class in range(len(MIoU)):
            pa = str(round(MPA[ind_class] * 100, 2)) if not np.isnan(MPA[ind_class]) else 'nan'
            iou = str(round(MIoU[ind_class] * 100, 2)) if not np.isnan(MIoU[ind_class]) else 'nan'
            pc = str(round(Precision[ind_class] * 100, 2)) if not np.isnan(Precision[ind_class]) else 'nan'
            cr = str(round(Class_ratio[ind_class] * 100, 2)) if not np.isnan(Class_ratio[ind_class]) else 'nan'
            pr = str(round(Pred_retio[ind_class] * 100, 2)) if not np.isnan(Pred_retio[ind_class]) else 'nan'
            print('===>' + name_classes[ind_class] + ':\t' + pa + '\t' + iou + '\t' + pc + '\t' + cr + '\t' + pr)

    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape

        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

from PIL import Image
from datasets.cityscapes_Dataset import label_colours, NUM_CLASSES
def inspect_decode_labels(pred, num_images=1, num_classes=NUM_CLASSES, 
        inspect_split=[0.9, 0.8, 0.7, 0.5, 0.0], inspect_ratio=[1.0, 0.8, 0.6, 0.3]):
    """Decode batch of segmentation masks.
    
    Args:
      pred: result of inference.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.data.cpu().numpy()
    n, c, h, w = pred.shape
    pred = pred.transpose([0, 2, 3, 1])
    if n < num_images: 
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (w, h))
      pixels = img.load()
      for j_, j in enumerate(pred[i, :, :, :]):
          for k_, k in enumerate(j):
              assert k.shape[0] == num_classes
              k_value = np.max(softmax(k))
              k_class = np.argmax(k)
              for it, iv in enumerate(inspect_split):
                  if k_value > iv: break
              if iv > 0:
                pixels[k_,j_] = tuple(map(lambda x: int(inspect_ratio[it]*x), label_colours[k_class]))
      outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)

def softmax(k, axis=None):
    exp_k = np.exp(k)
    return exp_k / np.sum(exp_k, axis=axis, keepdims=True)

