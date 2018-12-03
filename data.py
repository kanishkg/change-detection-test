import os
import torch

from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ChangeDetectionDataset(Dataset):

  def __init__(self, data_dir, transforms = None):
    self.data_dir = data_dir
    self.transforms = transforms
    class_1 = sorted(os.listdir(self.data_dir+'1/'))
    class_2 = sorted(os.listdir(self.data_dir+'0/'))
    self.dataset = []
    for i in range(len(class_1)/2):
      if i==0:
        print class_1[2*i],class_1[2*i+1]
      self.dataset.append({'a':class_1[2*i],
                           'b':class_1[2*i+1],
                           'class': 1})
    for i in range(len(class_2)/2):
      self.dataset.append({'a':class_2[2*i],
                           'b':class_2[2*i+1],
                           'class': 0})

    # for i in range(len(class_1)/2+len(class_2)/2-1):
    #   if "%05d" % (i+1)+"a.png" in class_1:
    #     self.dataset.append({'a':"%07d" % (i+1)+"a.png",
    #                          'b':"%07d" % (i+1)+"b.png",
    #                          'class': 1})
    #   else:
    #     self.dataset.append({'a':"%07d" % (i+1)+"a.png",
    #                          'b':"%07d" % (i+1)+"b.png",
    #                          'class': 0})

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    instance = self.dataset[idx]
    image_a = io.imread(self.data_dir+str(instance['class'])+'/'+instance['a'])[12:72,12:72,:]
    image_b = io.imread(self.data_dir+str(instance['class'])+'/'+instance['b'])[12:72,12:72,:]
    cl = instance['class']

    sample = {
          'a': image_a,
          'b': image_b,
          'c': cl
        }
    if self.transforms:
      sample['a'] = self.transforms(sample['a'])
      sample['b'] = self.transforms(sample['b'])

    return sample


class PreChangeDetectionDataset(Dataset):

  def __init__(self, data_dir, transforms = None):
    self.data_dir = data_dir
    self.transforms = transforms
    class_1 = os.listdir(self.data_dir+'1/')
    class_2 = os.listdir(self.data_dir+'0/')
    self.dataset = []
    for i in range(len(class_1)/2):
      if i==0:
        print class_1[2*i],class_1[2*i+1]
      self.dataset.append(data_dir+'1/'+class_1[2*i])
      self.dataset.append(data_dir+'1/'+class_1[2*i+1])

    for i in range(len(class_2)/2):
      self.dataset.append(data_dir+'0/'+class_2[2*i])
      self.dataset.append(data_dir+'0/'+class_2[2*i+1])

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    instance = self.dataset[idx]
    image = io.imread(instance)[12:76,12:76,:]

    if self.transforms:
      image = self.transforms(image)

    return  image


