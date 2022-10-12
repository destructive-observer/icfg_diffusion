# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Score SDE library
# which was released under the Apache License.
#
# Source:
# https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_Apache). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

# from fcntl import F_FULLFSYNC
from . import layers
from . import up_or_down_sampling, dense_layer
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from . import utils

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init
dense = dense_layer.dense
dense_params=dense_layer.dense_params
groupnorm_params = dense_layer.groupnorm_params
contract_inner = layers.contract_inner
ddpm_conv3x3_params = dense_layer.ddpm_conv3x3_params
ddpm_conv1x1_params = dense_layer.ddpm_conv1x1_params
up_down_conv2d_params=dense_layer.up_down_conv2d_params
upsample_conv_2d=up_or_down_sampling.upsample_conv_2d
dense_resnet_params = dense_layer.dense_resnet_params
conv_downsample_2d=up_or_down_sampling.conv_downsample_2d
class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups,in_channel, style_dim):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                   kernel=3, up=True,
                                                   resample_kernel=fir_kernel,
                                                   use_bias=True,
                                                   kernel_init=default_init())
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      if not self.with_conv:
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = self.Conv2d_0(x)

    return h


class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                   kernel=3, down=True,
                                                   resample_kernel=fir_kernel,
                                                   use_bias=True,
                                                   kernel_init=default_init())
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      if self.with_conv:
        x = F.pad(x, (0, 1, 0, 1))
        x = self.Conv_0(x)
      else:
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      if not self.with_conv:
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = self.Conv2d_0(x)

    return x


class ResnetBlockDDPMpp_Adagn(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0.):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
      
      
    self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None, zemb=None):
    h = self.act(self.GroupNorm_0(x, zemb))
    h = self.Conv_0(h)
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h, zemb))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp_Adagn(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
    
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)
   
    self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None, zemb=None):
    h = self.act(self.GroupNorm_0(x, zemb))

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h, zemb))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
   
    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
  

class ResnetBlockBigGANpp_Adagn_one(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
   
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)
    

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None, zemb=None):
    h = self.act(self.GroupNorm_0(x, zemb))

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
  
def ResnetBlockBigGANpp_ICFG(act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.,requires_grad=True):

  out_ch = out_ch if out_ch else in_ch
  p={'conv0':ddpm_conv3x3_params(in_ch, out_ch)}
  p['conv1']=ddpm_conv3x3_params(out_ch, out_ch, init_scale=init_scale)
  if temb_dim is not None:
    p['dense0']=dense_resnet_params(temb_dim,out_ch)
  if in_ch != out_ch or up or down:
    p['conv2']=ddpm_conv1x1_params(in_ch, out_ch)
  groupnorm0_function,p['groupnorm0'] = AdaptiveGroupNorm_ICFG(min(in_ch // 4, 32), in_ch, zemb_dim)
  groupnorm1_function,p['groupnorm1'] = AdaptiveGroupNorm_ICFG(min(in_ch // 4, 32), out_ch, zemb_dim)
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def f(x,params,base,temb,zemb):
    h = act(groupnorm0_function(x,params,base+'.groupnorm0',zemb))

    if up:
      if fir:
        h = up_or_down_sampling.upsample_2d(h, fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif down:
      if fir:
        h = up_or_down_sampling.downsample_2d(h, fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)
    # for i,value in params.items():
    #   print(i)
    h = F.conv2d(h,params.get(base+'.conv0.w'),params.get(base+'.conv0.b'), stride=1, padding=1)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += F.linear(act(temb),params.get(base+'.dense0.w'),params.get(base+'.dense0.b'))[:, :, None, None]
    h = act(groupnorm1_function(h,params,base+'.groupnorm1', zemb))
    h = F.dropout(h,dropout)
    h =  F.conv2d(h,params.get(base+'.conv1.w'),params.get(base+'.conv1.b'), stride=1, padding=1,
                   dilation=1)
    if in_ch != out_ch or up or down:
      x =  F.conv2d(x,params.get(base+'.conv2.w'),params.get(base+'.conv2.b'), stride=1, padding=0)
    # print('x.shape-{}, h.shape-{}'.format(x.shape,h.shape))
    if not skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
  return f,flat_params


def AdaptiveGroupNorm_ICFG(num_groups,in_channel, style_dim,requires_grad=True,eps=1e-5):
  def get_group_param(in_channel, style_dim):
    style = dense_params(style_dim, in_channel * 2)
    style['bias'][:in_channel]=1
    style['bias'][in_channel:]=0
    return {'norm':groupnorm_params(in_channel,False),'style':style}
  p={'adaptive':get_group_param(in_channel, style_dim)}
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def f(input,params,base,style):
    style = F.linear(style,params.get(base+'.adaptive.style.weight'),params.get(base+'.adaptive.style.bias')).unsqueeze(2).unsqueeze(3)
    gamma, beta = style.chunk(2,1)
    out = F.group_norm(
            input, num_groups, params.get(base+'.adaptive.norm.w'),  params.get(base+'.adaptive.norm.b'),eps)
    out = gamma * out + beta
    return out
  return f,flat_params

def AttnBlockpp_ICFG(channels, skip_rescale=False, init_scale=0.,requires_grad=True):
  def get_nin_param(in_dim, num_units, init_scale=0.1):
    return{'w':default_init(scale=init_scale)((in_dim, num_units)),'b':torch.zeros(num_units)}

  def f_nin(input,param,base):
    x = input.permute(0, 2, 3, 1)
    # for i,value in param.items():
    #   print(i)
    y = contract_inner(x, param.get(base+'.w')) + param.get(base+'.b')
    return y.permute(0, 3, 1, 2)
  def get_nin_group(channels):
    return {'nin0':get_nin_param(channels,channels),'nin1':get_nin_param(channels,channels)
    ,'nin2':get_nin_param(channels,channels),'nin3':get_nin_param(channels,channels, init_scale=init_scale)}
  p={'attn':get_nin_group(channels)}
  p['groupnorm'] = groupnorm_params(num_channels=channels)    
  flat_params = utils.cast(utils.flatten(p))
  # for i,value in flat_params.items():
  #   print(i)
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def f(x,params,base):
    B, C, H, W = x.shape
    h = F.group_norm(
            x, min(channels // 4, 32),params.get(base+'.groupnorm.w'),  params.get(base+'.groupnorm.b'),eps=1e-6)
    q = f_nin(h,params,base+'.attn.nin0')
    k = f_nin(h,params,base+'.attn.nin1')
    v = f_nin(h,params,base+'.attn.nin2')
    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = f_nin(h,params,base+'.attn.nin3')
    if not skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
  return f,flat_params

def Up_down_conv2d_ICFG(in_ch, out_ch, kernel,up=False, down=False,
               resample_kernel=(1, 3, 3, 1),use_bias=True,
               kernel_init=None,requires_grad=True):
    def get_conv2d_params(in_ch, out_ch, kernel,kernel_init=None):
      return{'conv0':up_down_conv2d_params(in_ch, out_ch, kernel,kernel_init)}
    p={'up_down_conv':get_conv2d_params(in_ch, out_ch, kernel,kernel_init)}
    flat_params = utils.cast(utils.flatten(p))
    if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
    def f(x,params,base):
      if up:
        # print(params.get(base+'.up_down_conv.conv0.w').shape)
        x = upsample_conv_2d(x, params.get(base+'.up_down_conv.conv0.w'), k=resample_kernel)
      elif down:
        x = conv_downsample_2d(x, params.get(base+'.up_down_conv.conv0.w'), k=resample_kernel)
      else:
        x = F.conv2d(x, params.get(base+'.up_down_conv.conv0.w'), stride=1, padding=kernel // 2)

      if use_bias:
        x = x + params.get(base+'.up_down_conv.conv0.b').reshape(1, -1, 1, 1)
      return x

    return f,flat_params

def Upsample_ICFG(in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1),requires_grad=True):
  def get_conv_params(in_ch, out_ch,with_conv,fir,fir_kernel=(1, 3, 3, 1)):
    up_function = None
    if fir:
      if with_conv:
        return{'conv':ddpm_conv3x3_params(in_ch,out_ch)}
    else:
      if with_conv:
        up_down = Up_down_conv2d_ICFG(in_ch,out_ch,kernel=3,up=True,resample_kernel=fir_kernel,use_bias=True,kernel_init=default_init())
        up_function = up_down[0]
        return{'conv':up_down[1]},up_function
  result = get_conv_params(in_ch, out_ch,with_conv,fir,fir_kernel)
  p={'upsample':result[0]}
  if result[1] is not None:
    up_function = get_conv_params(in_ch, out_ch,with_conv,fir,fir_kernel)[0]
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def f(x,params,base):
    B, C, H, W = x.shape
    if not fir:
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      if with_conv:
        h = F.conv2d(h,params.get(base+'.upsample.conv.w'),params.get(base+'.upsample.conv.b'),padding=1,stride=1,dilation=1)
    else:
      if not with_conv:
        h = up_or_down_sampling.upsample_2d(x, fir_kernel, factor=2)
      else:
        h = up_function(x,params,base+'.upsample.conv')

    return h
  return f,flat_params

def Downsample_ICFG(in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1),requires_grad=True):
  def get_conv_params(in_ch, out_ch,with_conv,fir,fir_kernel=(1, 3, 3, 1)):
    up_function = None
    if not fir:
      if with_conv:
        return{'conv_0':ddpm_conv3x3_params(in_ch,out_ch)}
    else:
      if with_conv:
        up_down = Up_down_conv2d_ICFG(in_ch,out_ch,kernel=3,down=True,resample_kernel=fir_kernel,use_bias=True,kernel_init=default_init())
        up_function = up_down[0]
        return{'conv_2d':up_down[1]},up_function
  result = get_conv_params(in_ch, out_ch,with_conv,fir,fir_kernel)
  p={'downsample':result[0]}
  if result[1] is not None:
    down_function = result[1]
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def f(x,params,base):
    if not fir:
      if with_conv:
        x = F.pad(x, (0, 1, 0, 1))
        x = F.conv2d(x,params.get(base+'.downsample.conv_0.w'),params.get(base+'.downsample.conv_0.b'),padding=0,stride=2)
      else:
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      if not with_conv:
        x = up_or_down_sampling.downsample_2d(x, fir_kernel, factor=2)
      else:
        x = down_function(x,params,base+'.downsample.conv_2d')
    return x
  return f,flat_params

def Combine_ICFG(dim1, dim2, method='cat',requires_grad=True):
  p={'combine':ddpm_conv1x1_params(dim1,dim2)}
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def f(x,params,base,y):
    h =  F.conv2d(x,params.get(base+'.combine.w'),params.get(base+'.combine.b'),padding=1,stride=1,dilation=1)
    if method == 'cat':
      return torch.cat([h, y], dim=1)
    elif method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {method} not recognized.')  
  return f,flat_params

def GaussianFourierProjection_ICFG(embedding_size=256, scale=1.0,requires_grad=False):
  p={'w':torch.randn(embedding_size) * scale}
  flat_params = utils.cast(utils.flatten(p))
  if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
  def f(x,params,base):
    x_proj = x[:, None] * params[base+'.w'][None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
  return f,flat_params