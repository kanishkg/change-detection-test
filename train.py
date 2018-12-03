import torch
import torch.nn as nn
from torchvision import transforms, datasets

from data import ChangeDetectionDataset

from models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
PATH = '/data/kvg245/change_test/checkpoints/train.ckpt'
print "Loading data"
change_detection_dataset = ChangeDetectionDataset(data_dir='/data/kvg245/',
                                                  transforms = transforms.Compose([
                                                    transforms.ToTensor()
                                                    ]))
num_epochs = 20
batch_size = 32
learning_rate = 0.001
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

print "Building model"
model = Siamese().to(device)


print model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  for i, sample in enumerate(train_loader):
    optimizer.zero_grad()
    sample['a'] = sample['a'].cuda()
    sample['b'] = sample['b'].cuda()
    sample['c'] = sample['c'].cuda()
    outputs = model(sample['a'], sample['b'])
    loss = criterion(outputs, sample['c'])
    loss.backward()
    optimizer.step()

    if i%50 == 0:
      # print outputs, sample['c']
      print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
           % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

  correct = 0
  total = 0

  for sample in val_loader:
    sample['a'] = sample['a'].cuda()
    sample['b'] = sample['b'].cuda()
    sample['c'] = sample['c'].cuda()
 
    outputs = model(sample['a'], sample['b'])
    _, predicted = torch.max(outputs.data, 1)
    total += sample['c'].size(0)
    correct += (predicted == sample['c']).sum()
  print('Accuracy of the model on the val images: %d %%' % (100 * correct / total))
  if (epoch+1)%5 == 0:
    torch.save(model, PATH)
correct = 0
total = 0

for sample in test_loader:
  sample['a'] = sample['a'].cuda()
  sample['b'] = sample['b'].cuda()
  sample['c'] = sample['c'].cuda()
 
  outputs = model(sample['a'], sample['b'])
  _, predicted = torch.max(outputs.data, 1)
  total += sample['c'].size(0)
  correct += (predicted == sample['c']).sum()
print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))






