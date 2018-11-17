import torch
import torchvision
import torch.nn as nn


base_conv_cfg = {
    'num_layers':2,
    'inp': [3,16],
    'out': [16,32],
    'stride': [4,2],
    'kernel': [8,4],
    'pool': [False,False]}

class ConvBlock(nn.Module):

  def __init__(self, cfg):
    super(ConvBlock, self).__init__()

    self.layers = nn.ModuleList([])
    for l in range(cfg['num_layers']):
      self.layers += [nn.Conv2d(
        cfg['inp'][l],
        cfg['out'][l],
        cfg['kernel'][l],
        cfg['stride'][l]
        ), nn.BatchNorm2d(cfg['out'][l]), nn.ReLU()]

  def forward(self,x):
    for layer in self.layers:
       x = layer(x)
    return x

class SimpleLSTM(nn.Module):

  def __init__(self, hidden_size=256, cell_type='lstm'):
    super(ConvBlock, self).__init__()

    self.conv_block = ConvBlock(base_conv_cfg)
    self.hidden_size = hidden_size
    if cell_type=='lstm':
      self.rnn = nn.LSTM(2592, hidden_size)
    elif cell_type=='gru':
      self.rnn = nn.GRU(2592, hidden_size)
    self.fc = nn.Linear(256, 2)

  def forward(self, x, y):
    x = self.conv_block(x)
    y = self.conv_block(y)
    x = x.view(-1,2592)
    y = y.view(-1,2592)
    seq = torch.cat(torch.t(x),torch.t(y),0)
    out, _ = self.rnn(seq)[-1,:,:]
    out = nn.softmax(self.fc(out))
    return out

