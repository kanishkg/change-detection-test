import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

base_conv_cfg = {
    'num_layers':2,
    'inp': [3,16],
    'out': [16,32],
    'stride': [4,2],
    'kernel': [8,4],
    'pool': [False,False]}

alt_base_conv_cfg = {
    'num_layers':4,
    'inp': [3, 16, 32, 32],
    'out': [16, 32, 32, 64],
    'stride': [1,2,1,2],
    'kernel': [4,4,2,2],
    'pool': [False,False,False,False]}

alt = {
    'num_layers':5,
    'inp': [3, 32, 32, 64, 64],
    'out': [32, 32, 64, 64, 256],
    'stride': [2,2,2,2,1],
    'kernel': [4,4,4,4,4],
    'padding':[1,1,1,1,0],
    'pool': [False,False,False,False,False]}
decon_alt = {
    'num_layers':5,
    'inp': [3, 32, 32, 64, 64],
    'out': [32, 32, 64, 64, 256],
    'stride': [2,2,2,2,1],
    'kernel': [4,4,4,4,4],
    'padding':[1,1,1,1,0],
    'pool': [False,False,False,False,False]}

big_conv_cfg = {
    'num_layers':4,
    'inp': [3, 16, 32, 32],
    'out': [16, 32, 32, 64],
    'stride': [2,2,2,2],
    'kernel': [8,4,2,2],
    'padding': [0, 1, 1, 1],
    'pool': [False,False,False,False]}




stack_conv_cfg = {
    'num_layers':2,
    'inp': [6,32],
    'out': [32, 64],
    'stride': [4,2],
    'kernel': [8,4],
    'pool': [False,False]}



class DeConvBlock(nn.Module):

  def __init__(self, cfg):
    super(DeConvBlock, self).__init__()

    self.layers = nn.ModuleList([])

    for l in reversed(range(cfg['num_layers'])):
      self.layers += [nn.ConvTranspose2d(
        cfg['out'][l],
        cfg['inp'][l],
        cfg['kernel'][l],
        cfg['stride'][l],
        cfg['padding'][l]
        ), nn.ReLU()]

  def forward(self,x):
    for layer in self.layers:
      x = layer(x)
    #J  print x.shape

    return x

class ConvBlock(nn.Module):

  def __init__(self, cfg):
    super(ConvBlock, self).__init__()

    self.layers = nn.ModuleList([])
    for l in range(cfg['num_layers']):
      self.layers += [nn.Conv2d(
        cfg['inp'][l],
        cfg['out'][l],
        cfg['kernel'][l],
        cfg['stride'][l],
        cfg['padding'][l]
        ), nn.ReLU()]

  def forward(self,x):
    for layer in self.layers:
      x = layer(x)
    #  print x.shape
    return x

class MiniVAE(nn.Module):
  def __init__(self, hidden_size =256):
    super(MiniVAE, self).__init__()

    self.hidden_size = hidden_size
    self.conv_block = ConvBlock(alt)
    self.fc_enc_m = nn.Linear(hidden_size, hidden_size/2)
    self.fc_enc_s = nn.Linear(hidden_size, hidden_size/2)

    self.fc_dec = nn.Linear(hidden_size/2, hidden_size)
    self.deconv_block = DeConvBlock(decon_alt)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

  def encode(self, x):
    x = self.conv_block(x)
    x = x.view(-1,256)
    mu = F.relu(self.fc_enc_m(x))
    logvar = F.relu(self.fc_enc_s(x))
    return mu, logvar
    
  def decode(self, z):
    y = F.relu(self.fc_dec(z))
    y = y.view(-1, 256, 1, 1)
    y = self.deconv_block(y)
    return y

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    y = self.decode(z)
    return y, mu, logvar

class SimpleLSTM(nn.Module):

  def __init__(self, hidden_size=256, cell_type='lstm'):
    super(SimpleLSTM, self).__init__()

    self.conv_block = ConvBlock(base_conv_cfg)
    self.hidden_size = hidden_size
    self.fc1 = nn.Linear(2592,hidden_size)
    if cell_type=='lstm':
      self.rnn = nn.LSTM(hidden_size, hidden_size)
    elif cell_type=='gru':
      self.rnn = nn.GRU(hidden_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 2)

  def forward(self, x, y):
    x = self.conv_block(x)
    y = self.conv_block(y)
    x = x.view(-1,2592).unsqueeze(0)
    y = y.view(-1,2592).unsqueeze(0)
    x = F.relu(self.fc1(x))
    y = F.relu(self.fc1(y))
    seq = torch.cat([x,y],0)
    out, _ = self.rnn(seq)
    out = self.fc(out[-1,:,:])
    return out

class Siamese(nn.Module):

  def __init__(self, hidden_size=256):
    super(Siamese, self).__init__()

    self.conv_block = ConvBlock(base_conv_cfg)
    self.hidden_size = hidden_size
    self.fc1 = nn.Linear(1152*2, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 2)

  def forward(self, x, y):
    x = self.conv_block(x)
    y = self.conv_block(y)
    x = x.view(-1,1152)
    y = y.view(-1,1152)
    out = torch.cat([x,y],1)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out

class Diffnet(nn.Module):

  def __init__(self, hidden_size=256):
    super(Diffnet, self).__init__()

    self.conv_block = ConvBlock(base_conv_cfg)
    self.hidden_size = hidden_size
    self.fc1 = nn.Linear(2592, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 2)

  def forward(self, x, y):
    out = self.conv_block(x-y)
    out = out.view(-1,2592)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out

class SiameseDiffnet(nn.Module):

  def __init__(self, hidden_size=256, mode='dif'):
    super(SiameseDiffnet, self).__init__()

    self.conv_block = ConvBlock(base_conv_cfg)
    self.hidden_size = hidden_size
    self.fc1 = nn.Linear(2592, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 2)
    self.mode = mode

  def forward(self, x, y):
    x = self.conv_block(x)
    y = self.conv_block(y)
    x = x.view(-1,2592)
    y = y.view(-1,2592)
    if self.mode == 'dif':
      out = x-y
    elif self.mode == 'dot':
      out = torch.mul(x, y)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out

class Stacknet(nn.Module):

  def __init__(self, hidden_size=256, mode='dif'):
    super(Stacknet, self).__init__()

    self.conv_block = ConvBlock(stack_conv_cfg)
    self.hidden_size = hidden_size
    self.fc1 = nn.Linear(5184, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 2)
    self.mode = mode

  def forward(self, x, y):
    out = torch.cat([x,y],1)
    out = self.conv_block(out)
    out = out.view(-1,5184)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out

class SiameseOpnet(nn.Module):

  def __init__(self, hidden_size=256, mode='dif'):
    super(SiameseOpnet, self).__init__()

    self.conv_block = ConvBlock(base_conv_cfg)
    self.hidden_size = hidden_size
    self.fc1 = nn.Linear(2592, hidden_size)
    # self.c1 = nn.Conv2d(2,1,1)
    self.w = nn.Parameter(torch.randn(1))
    self.v = nn.Parameter(torch.randn(1))
    # self.w = 1
    # self.v = 1




    self.fc2 = nn.Linear(hidden_size, 2)

  def forward(self, x, y):
    x = self.conv_block(x)
    y = self.conv_block(y)
    x = x.view(-1,2592)
    y = y.view(-1,2592)

    out = self.w*x-self.v*y
    # out = torch.stack([x,y],dim=1).unsqueeze(3)
    # out = self.c1(out)
    # out = out.view(-1,self.hidden_size)

    out = F.relu(self.fc1(out))
    # y = F.relu(self.fc1(y))
    out = self.fc2(out)
    return out


