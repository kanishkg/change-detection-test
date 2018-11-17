import torch
import torch.nn as nn
from torchvision import transforms, datasets

from data import ChangeDetectionDataset

from models import *


change_detection_dataset = ChangeDetectionDataset(data_dir='/data/kvg245/',
                                                  transforms = transforms.Compose([
                                                    transforms.ToTensor()
                                                    ]))
num_epochs = 20
batch_size = 32
train_size = 0.8 * len(change_detection_dataset)
val_size = 0.1 * len(change_detection_dataset)
test_size = len(change_detection_dataset)-tarin_size-val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=batch_size, shuffle=False, num_workers=4)

model = SimpleLSTM()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  for i, (image_a,image_b, labels) in enumerate(train_loader):
    optimizer.zero_grad() 
    outputs = model(image_a, image_b)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if i%50 == 0:
      print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
           % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

  correct = 0
  total = 0

  for image_a, image_b, labels in val_loader:
    outputs = model(image_a, image_b)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    print('Accuracy of the model on the val images: %d %%' % (100 * correct / total))
  if (epoch+1)%5 == 0:
    torch.save({
         'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'loss': loss,
         'val_acc': 100*correct/total
         }, '/data/kvg245/change_test')

correct = 0
total = 0
for image_a, image_b, labels in test_loader:
  outputs = model(image_a, image_b)
  _, predicted = torch.max(outputs.data, 1)
  total += labels.size(0)
  correct += (predicted == labels).sum()
  print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
