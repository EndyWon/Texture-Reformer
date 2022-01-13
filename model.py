# borrow from https://github.com/MingSun-Tse/Collaborative-Distillation/blob/master/model/model_original.py

import numpy as np
import torch.nn as nn
import torch

# Original VGG19
# Encoder1/Decoder1
class Encoder1(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder1, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv11 = nn.Conv2d(  3, 64,3,1,0, dilation=1)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False

  def forward(self, input):
    y = self.conv0(input)
    y = self.relu(self.conv11(self.pad(y)))
    return y


class Decoder1(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder1, self).__init__()
    self.fixed = fixed
    
    self.conv11 = nn.Conv2d( 64,  3,3,1,0, dilation=1)
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  def forward(self, input):
    y = self.relu(self.conv11(self.pad(input)))
    return y
  

# Encoder2/Decoder2
class Encoder2(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder2, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv11 = nn.Conv2d(  3, 64,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 64,128,3,1,0)
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    y = self.conv0(input)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    return y


class Decoder2(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder2, self).__init__()
    self.fixed = fixed
    
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    y = self.relu(self.conv21(self.pad(input)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y

    
# Encoder3/Decoder3
class Encoder3(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder3, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv11 = nn.Conv2d(  3, 64,3,1,0) # conv1_1
    self.conv12 = nn.Conv2d( 64, 64,3,1,0) # conv1_2
    self.conv21 = nn.Conv2d( 64,128,3,1,0) # conv2_1
    self.conv22 = nn.Conv2d(128,128,3,1,0) # conv2_2
    self.conv31 = nn.Conv2d(128,256,3,1,0) # conv3_1
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    y = self.conv0(input)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y)))
    return y


class Decoder3(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder3, self).__init__()
    self.fixed = fixed
    
    self.conv31 = nn.Conv2d(256,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    y = self.relu(self.conv31(self.pad(input)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y


# Encoder4/Decoder4
class Encoder4(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder4, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv11 = nn.Conv2d(  3, 64,3,1,0) # conv1_1
    self.conv12 = nn.Conv2d( 64, 64,3,1,0) # conv1_2
    self.conv21 = nn.Conv2d( 64,128,3,1,0) # conv2_1
    self.conv22 = nn.Conv2d(128,128,3,1,0) # conv2_2
    self.conv31 = nn.Conv2d(128,256,3,1,0) # conv3_1
    self.conv32 = nn.Conv2d(256,256,3,1,0) # conv3_2
    self.conv33 = nn.Conv2d(256,256,3,1,0) # conv3_3
    self.conv34 = nn.Conv2d(256,256,3,1,0) # conv3_4
    self.conv41 = nn.Conv2d(256,512,3,1,0) # conv4_1
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    y = self.conv0(input)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv34(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv41(self.pad(y)))
    return y
  
    
class Decoder4(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder4, self).__init__()
    self.fixed = fixed

    self.conv41 = nn.Conv2d(512,256,3,1,0)
    self.conv34 = nn.Conv2d(256,256,3,1,0)
    self.conv33 = nn.Conv2d(256,256,3,1,0)
    self.conv32 = nn.Conv2d(256,256,3,1,0)
    self.conv31 = nn.Conv2d(256,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    y = self.relu(self.conv41(self.pad(input)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y
  
    
# Encoder5/Decoder5
class Encoder5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder5, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
                                    [[[[0]],[[0]],[[255]]],
                                     [[[0]],[[255]],[[0]]],
                                     [[[255]],[[0]],[[0]]]])).float())
    self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
                                    [-103.939,-116.779,-123.68])).float())
    self.conv11 = nn.Conv2d(  3, 64,3,1,0) # conv1_1
    self.conv12 = nn.Conv2d( 64, 64,3,1,0) # conv1_2
    self.conv21 = nn.Conv2d( 64,128,3,1,0) # conv2_1
    self.conv22 = nn.Conv2d(128,128,3,1,0) # conv2_2
    self.conv31 = nn.Conv2d(128,256,3,1,0) # conv3_1
    self.conv32 = nn.Conv2d(256,256,3,1,0) # conv3_2
    self.conv33 = nn.Conv2d(256,256,3,1,0) # conv3_3
    self.conv34 = nn.Conv2d(256,256,3,1,0) # conv3_4
    self.conv41 = nn.Conv2d(256,512,3,1,0) # conv4_1
    self.conv42 = nn.Conv2d(512,512,3,1,0) # conv4_2
    self.conv43 = nn.Conv2d(512,512,3,1,0) # conv4_3
    self.conv44 = nn.Conv2d(512,512,3,1,0) # conv4_4
    self.conv51 = nn.Conv2d(512,512,3,1,0) # conv5_1
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    y = self.conv0(input)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv34(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv41(self.pad(y)))
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv43(self.pad(y)))
    y = self.relu(self.conv44(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv51(self.pad(y)))
    return y


class Decoder5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder5, self).__init__()
    self.fixed = fixed
    
    self.conv51 = nn.Conv2d(512,512,3,1,0)
    self.conv44 = nn.Conv2d(512,512,3,1,0)
    self.conv43 = nn.Conv2d(512,512,3,1,0)
    self.conv42 = nn.Conv2d(512,512,3,1,0)
    self.conv41 = nn.Conv2d(512,256,3,1,0)
    self.conv34 = nn.Conv2d(256,256,3,1,0)
    self.conv33 = nn.Conv2d(256,256,3,1,0)
    self.conv32 = nn.Conv2d(256,256,3,1,0)
    self.conv31 = nn.Conv2d(256,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    y = self.relu(self.conv51(self.pad(input)))
    y = self.unpool(y)
    y = self.relu(self.conv44(self.pad(y)))
    y = self.relu(self.conv43(self.pad(y)))
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y

