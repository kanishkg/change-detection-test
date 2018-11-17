import os
import torch

from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ChangeDetectionDataset(Dataset):

  def __init__(self, data_dir, transforms = None):
    self.data_dir = data_dir
    self.transforms = transform
    class_1 = os.listdir(self.data_dir+'1/')
    class_2 = os.listdir(self.data_dir+'0/')
    self.dataset = []

    for i in range(len(class_1)/2+len(class_2)/2):
      if "%05d" % (i+1)+"a.png" in class_1:
        self.dataset.append({'a':"%05d" % (i+1)+"a.png",
                             'b':"%05d" % (i+1)+"b.png",
                             'class': 1})
      else:
        self.dataset.append({'a':"%05d" % (i+1)+"a.png",
                             'b':"%05d" % (i+1)+"b.png",
                             'class': 0})

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    instance = self.dataset[idx]
    image_a = io.imread(self.data_dir+str(instance['class'])+'/'+instance['a'])
    image_b = io.imread(self.data_dir+str(instance['class'])+'/'+instance['b'])
    cl = instance['class']

    sample = {
          'a': image_a,
          'b': image_b,
          'c': cl
        }
    if self.transform:
      sample = self.transform(sample)

    return sample


