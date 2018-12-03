import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.nn import functional as F

from data import PreChangeDetectionDataset
from models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
PATH = '/data/kvg245/change_test/checkpoints/pretrain_vaeeee.ckpt'
change_detection_dataset = PreChangeDetectionDataset(data_dir='/data/kvg245/',
                                                  transforms = transforms.Compose([
                                                    transforms.ToTensor()
                                                    ]))
num_epochs = 200
ckpt = True
batch_size = 64
learning_rate = 0.0001
train_size = int(0.8 * len(change_detection_dataset))
val_size = int(0.1 * len(change_detection_dataset))
test_size = len(change_detection_dataset)-train_size-val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(change_detection_dataset, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=batch_size, shuffle=False, num_workers=4)

if ckpt:
  model = torch.load(PATH).to(device)
else:
  model = MiniVAE().to(device)

print model

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def kl_loss(mu, logvar):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return KLD

criterion = nn.MSELoss()
for epoch in range(num_epochs):
  for i, image in enumerate(train_loader):
    optimizer.zero_grad()
    image = image.cuda()
    y, mu, logvar = model(image)
    kl = kl_loss(mu, logvar)
    l2 = criterion(y, image)
    loss =  l2+kl
    loss.backward()
    optimizer.step()

    if i%50 == 0:
      # print outputs, sample['c']
      print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, l2: %.4f,kl:%.4f' 
           % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0],l2.data[0],kl.data[0]))
  if (epoch+1)%2 == 0:
    torch.save(model, PATH)


