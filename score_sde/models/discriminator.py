# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
from email.policy import strict
import torch
import torch.nn as nn
import numpy as np


from . import up_or_down_sampling
from . import dense_layer
from . import layers
from . import utils
import torch.nn.functional as F

dense = dense_layer.dense
conv2d = dense_layer.conv2d
dense_params=dense_layer.dense_params
conv2d_params=dense_layer.conv2d_params
conv2dT_params=dense_layer.conv2dT_params
get_sinusoidal_positional_embedding = layers.get_timestep_embedding


def TimestepEmbedding_ICFG(embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2),requires_grad=True):
    def get_dense_params(embedding_dim, hidden_dim, output_dim):
        return{
            'dense0':dense_params(embedding_dim,hidden_dim),
            'dense1':dense_params(hidden_dim,output_dim)
        }
    act_f = act
    p={'dense':get_dense_params(embedding_dim, hidden_dim, output_dim)}
    flat_params = utils.cast(utils.flatten(p))
    if requires_grad:
        utils.set_requires_grad_except_bn_(flat_params)
    def f(temp,params,base):
        # print(temp)
        temb = get_sinusoidal_positional_embedding(temp, embedding_dim)
        # print(temb)
        temb = F.linear(temb, params[base+'.dense.dense0.weight'], params[base+'.dense.dense0.bias'])
        # print(params[base+'.dense.dense0.weight'])
        # print(temb)
        temb = act_f(temb)
        # print(temb)
        temb = F.linear(temb, params[base+'.dense.dense1.weight'], params[base+'.dense.dense1.bias'])
        # temb = self.main(temb)
        # print(temb)
        return temb
    return f,flat_params

def  Discriminator_small_icfg(nc = 3, ngf = 64,kernel_size=3, t_emb_dim = 128, act=nn.LeakyReLU(0.2), padding=1,downsample=False
,do_bias=True,fir_kernel=(1,3,3,1),count=4,requires_grad=True):
    act_f = act
    
    def down_conv_params(in_channel,
        out_channel,
        kernel_size=3,
        t_emb_dim=128):
        return {'conv1':conv2d_params(in_channel,out_channel,kernel_size,do_bias),'conv2':conv2d_params(out_channel,out_channel,kernel_size,do_bias,init_scale=0.),'dense_t1':
        dense_params(t_emb_dim,out_channel),'skip':conv2d_params(in_channel,out_channel,1,do_bias=False)}
    def group_params(in_channel,
        out_channel,
        kernel_size=3,
        t_emb_dim=128):
        return{'group0':down_conv_params(in_channel*2,out_channel*2,kernel_size,t_emb_dim),
        'group1':down_conv_params(in_channel*2,out_channel*4,kernel_size,t_emb_dim),
        'group2':down_conv_params(in_channel*4,out_channel*8,kernel_size,t_emb_dim),
        'group3':down_conv_params(in_channel*8,out_channel*8,kernel_size,t_emb_dim)}

    p={'group_params':group_params(ngf,ngf,kernel_size,t_emb_dim)}
    t_f = TimestepEmbedding_ICFG(t_emb_dim,t_emb_dim,t_emb_dim,act=nn.LeakyReLU(0.2))
    p['t_emb']=t_f[1]
    # print(t_f[1]['dense.dense0.weight'])
    t_emb_function = t_f[0]
    # print(t_emb_function)
    p['final_conv']= conv2d_params(ngf*8+1, ngf*8, kernel_size,do_bias,init_scale=0.)
    p['start_conv']= conv2d_params(nc,ngf*2,1,do_bias,init_scale=0.)
    p['end_linear']= dense_params(ngf*8,1,init_scale=0.)
    ## here is different from ddgan's discriminatro Init_scale-0.

    flat_params = utils.cast(utils.flatten(p))
    if requires_grad:
        utils.set_requires_grad_except_bn_(flat_params)
    def run_down_conv(input,params,t_emb,base,downsample):
        out = act_f(input)
        out = F.conv2d(out,params[base+'.conv1.w'],params[base+'.conv1.b'],padding=padding,stride=1)
        # print(out.shape)
        # print(t_emb.shape)
        # print(params[base+'.dense_t1.weight'].shape)
        # print(params[base+'.dense_t1.bias'].shape)
        out += F.linear(t_emb,params[base+'.dense_t1.weight'],params[base+'.dense_t1.bias'])[..., None, None]
        out = act_f(out)
        if downsample:
            out = up_or_down_sampling.downsample_2d(out, fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, fir_kernel, factor=2)
        out = F.conv2d(out,params[base+'.conv2.w'],params[base+'.conv2.b'],padding=padding,stride=1)
        skip = F.conv2d(input,params[base+'.skip.w'],params.get(base+'skip.b'),padding=0,stride=1)
        out = (out + skip)/np.sqrt(2)
        return out
    def run_group(o,params,t_emb,downsample):
        o1 = run_down_conv(o ,params,t_emb,'group_params.group0',downsample=False)
        o2 = run_down_conv(o1,params,t_emb,'group_params.group1',downsample=True) 
        o3 = run_down_conv(o2,params,t_emb,'group_params.group2',downsample=True) 
        o4 = run_down_conv(o3,params,t_emb,'group_params.group3',downsample=True)            
        return o4
    def f(x,params,t_emb,x_t):
        # for k,v in sorted(params.items()):
        #     print('k name {}'.format(k))
        # print(t_emb)
        t_emb_result = act_f(t_emb_function(t_emb,params,'t_emb'))
        # print(t_emb_result)
        input_x = torch.cat((x,x_t),dim=1)
        h0 = F.conv2d(input_x,params['start_conv.w'],params['start_conv.b'],padding=0,stride=1)
        out = run_group(h0,params,t_emb_result,downsample)

        batch, channel, height, width = out.shape
            
        stddev_group = 4
        stddev_feat = 1
    
        group = min(batch, stddev_group)
        stddev = out.view(
            group, -1,stddev_feat, channel //stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        # print('#######out###########{}'.format(out))
        out = F.conv2d(out,params['final_conv.w'],params['final_conv.b'],padding=1)
        # print(out)
        out = act_f(out)
   
        # print('#######out###########{}'.format(out.shape))
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        # print('#######out###########{}'.format(out))
        out = F.linear(out,params['end_linear.weight'],params['end_linear.bias'])
        # print('#######end_weight###########{}'.format(params['end_linear.weight']))
        # print(out)
        
        return out

    return f,flat_params


def  Discriminator_mid_icfg(nc = 3, ngf = 64,kernel_size=3, t_emb_dim = 128, act=nn.LeakyReLU(0.2), padding=1,downsample=False
,do_bias=True,fir_kernel=(1,3,3,1),count=4,requires_grad=True):
    act_f = act
    
    def down_conv_params(in_channel,
        out_channel,
        kernel_size=3,
        t_emb_dim=128):
        return {'conv1':conv2d_params(in_channel,out_channel,kernel_size,do_bias),'conv2':conv2d_params(out_channel,out_channel,kernel_size,do_bias,init_scale=0.),'dense_t1':
        dense_params(t_emb_dim,out_channel),'skip':conv2d_params(in_channel,out_channel,1,do_bias=False)}
    def group_params(in_channel,
        out_channel,
        kernel_size=3,
        t_emb_dim=128):
        return{'group0':down_conv_params(in_channel*2,out_channel*2,kernel_size,t_emb_dim),
        'group1':down_conv_params(in_channel*2,out_channel*4,kernel_size,t_emb_dim),
        'group2':down_conv_params(in_channel*4,out_channel*8,kernel_size,t_emb_dim),
        'group3':down_conv_params(in_channel*8,out_channel*8,kernel_size,t_emb_dim),
        'group4':down_conv_params(in_channel*8,out_channel*8,kernel_size,t_emb_dim)}

    p={'group_params':group_params(ngf,ngf,kernel_size,t_emb_dim)}
    t_f = TimestepEmbedding_ICFG(t_emb_dim,t_emb_dim,t_emb_dim,act=nn.LeakyReLU(0.2))
    p['t_emb']=t_f[1]
    # print(t_f[1]['dense.dense0.weight'])
    t_emb_function = t_f[0]
    # print(t_emb_function)
    p['final_conv']= conv2d_params(ngf*8+1, ngf*8, kernel_size,do_bias,init_scale=0.)
    p['start_conv']= conv2d_params(nc,ngf*2,1,do_bias,init_scale=0.)
    p['end_linear']= dense_params(ngf*8,1,init_scale=0.)
    ## here is different from ddgan's discriminatro Init_scale-0.

    flat_params = utils.cast(utils.flatten(p))
    if requires_grad:
        utils.set_requires_grad_except_bn_(flat_params)
    def run_down_conv(input,params,t_emb,base,downsample):
        out = act_f(input)
        out = F.conv2d(out,params[base+'.conv1.w'],params[base+'.conv1.b'],padding=padding,stride=1)
        # print(out.shape)
        # print(t_emb.shape)
        # print(params[base+'.dense_t1.weight'].shape)
        # print(params[base+'.dense_t1.bias'].shape)
        out += F.linear(t_emb,params[base+'.dense_t1.weight'],params[base+'.dense_t1.bias'])[..., None, None]
        out = act_f(out)
        if downsample:
            out = up_or_down_sampling.downsample_2d(out, fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, fir_kernel, factor=2)
        out = F.conv2d(out,params[base+'.conv2.w'],params[base+'.conv2.b'],padding=padding,stride=1)
        skip = F.conv2d(input,params[base+'.skip.w'],params.get(base+'skip.b'),padding=0,stride=1)
        out = (out + skip)/np.sqrt(2)
        return out
    def run_group(o,params,t_emb,downsample):
        o1 = run_down_conv(o ,params,t_emb,'group_params.group0',downsample=False)
        o2 = run_down_conv(o1,params,t_emb,'group_params.group1',downsample=True) 
        o3 = run_down_conv(o2,params,t_emb,'group_params.group2',downsample=True) 
        o4 = run_down_conv(o3,params,t_emb,'group_params.group3',downsample=True)  
        o5 = run_down_conv(o4,params,t_emb,'group_params.group3',downsample=True)          
        return o4
    def f(x,params,t_emb,x_t):
        # for k,v in sorted(params.items()):
        #     print('k name {}'.format(k))
        # print(t_emb)
        t_emb_result = act_f(t_emb_function(t_emb,params,'t_emb'))
        # print(t_emb_result)
        input_x = torch.cat((x,x_t),dim=1)
        h0 = F.conv2d(input_x,params['start_conv.w'],params['start_conv.b'],padding=0,stride=1)
        out = run_group(h0,params,t_emb_result,downsample)

        batch, channel, height, width = out.shape
            
        stddev_group = 4
        stddev_feat = 1
    
        group = min(batch, stddev_group)
        stddev = out.view(
            group, -1,stddev_feat, channel //stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        # print('#######out###########{}'.format(out))
        out = F.conv2d(out,params['final_conv.w'],params['final_conv.b'],padding=1)
        # print(out)
        out = act_f(out)
   
        # print('#######out###########{}'.format(out.shape))
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        # print('#######out###########{}'.format(out))
        out = F.linear(out,params['end_linear.weight'],params['end_linear.bias'])
        # print('#######end_weight###########{}'.format(params['end_linear.weight']))
        # print(out)
        
        return out

    return f,flat_params

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb

#%%
class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim = 128,
        downsample=False,
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
     
        
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        
        self.conv1 = nn.Sequential(
                    conv2d(in_channel, out_channel, kernel_size, padding=padding),
                    )

        
        self.conv2 = nn.Sequential(
                    conv2d(out_channel, out_channel, kernel_size, padding=padding,init_scale=0.)
                    )
        self.dense_t1= dense(t_emb_dim, out_channel)


        self.act = act
        
            
        self.skip = nn.Sequential(
                    conv2d(in_channel, out_channel, 1, padding=0, bias=False),
                    )
        
            

    def forward(self, input, t_emb):
        
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]
       
        out = self.act(out)
       
        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)
        
        
        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)


        return out

class Discriminator_small(nn.Module):
  """A time-dependent discriminator for small images (CIFAR10, StackMNIST)."""

  def __init__(self, nc = 3, ngf = 64, t_emb_dim = 128, act=nn.LeakyReLU(0.2)):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.act = act
    
    
    self.t_embed = TimestepEmbedding(
        embedding_dim=t_emb_dim,
        hidden_dim=t_emb_dim,
        output_dim=t_emb_dim,
        act=act,
        )
    
    
     
    # Encoding layers where the resolution decreases
    self.start_conv = conv2d(nc,ngf*2,1, padding=0, init_scale=0.)
    self.conv1 = DownConvBlock(ngf*2, ngf*2, t_emb_dim = t_emb_dim,act=act)
    
    self.conv2 = DownConvBlock(ngf*2, ngf*4,  t_emb_dim = t_emb_dim, downsample=True,act=act)
    
    
    self.conv3 = DownConvBlock(ngf*4, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act)

    
    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act)
    
    
    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1, init_scale=0.)
    self.end_linear = dense(ngf*8, 1, init_scale=0.)
    
    self.stddev_group = 4
    self.stddev_feat = 1
    
        
  def forward(self, x, t, x_t):
    t_embed = self.act(self.t_embed(t))  
    # print(t_embed)
  
    input_x = torch.cat((x, x_t), dim = 1)
    
    h0 = self.start_conv(input_x)
    h1 = self.conv1(h0,t_embed)    
    
    h2 = self.conv2(h1,t_embed)   
   
    h3 = self.conv3(h2,t_embed)
   
    
    out = self.conv4(h3,t_embed)
    
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([out, stddev], 1)
    
    out = self.final_conv(out)
    out = self.act(out)
   
    
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)
    
    return out


class Discriminator_large(nn.Module):
  """A time-dependent discriminator for large images (CelebA, LSUN)."""

  def __init__(self, nc = 1, ngf = 32, t_emb_dim = 128, act=nn.LeakyReLU(0.2)):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.act = act
    
    self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
      
    self.start_conv = conv2d(nc,ngf*2,1, padding=0)
    self.conv1 = DownConvBlock(ngf*2, ngf*4, t_emb_dim = t_emb_dim, downsample = True, act=act)
    
    self.conv2 = DownConvBlock(ngf*4, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act)

    self.conv3 = DownConvBlock(ngf*8, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act)
    
    
    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act)
    self.conv5 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act)
    self.conv6 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act)

  
    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1)
    self.end_linear = dense(ngf*8, 1)
    
    self.stddev_group = 4
    self.stddev_feat = 1
    
        
  def forward(self, x, t, x_t):
    t_embed = self.act(self.t_embed(t))  
    
    input_x = torch.cat((x, x_t), dim = 1)
    
    h = self.start_conv(input_x)
    h = self.conv1(h,t_embed)    
   
    h = self.conv2(h,t_embed)
   
    h = self.conv3(h,t_embed)
    h = self.conv4(h,t_embed)
    h = self.conv5(h,t_embed)
   
    
    out = self.conv6(h,t_embed)
    
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([out, stddev], 1)
    
    out = self.final_conv(out)
    out = self.act(out)
    
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)
    
    return out


def  Discriminator_large_icfg(nc = 3, ngf = 64,kernel_size=3, t_emb_dim = 128, act=nn.LeakyReLU(0.2), padding=1,downsample=False
,do_bias=True,fir_kernel=(1,3,3,1),count=6,requires_grad=True):
    act_f = act
    
    def down_conv_params(in_channel,
        out_channel,
        kernel_size=3,
        t_emb_dim=128):
        return {'conv1':conv2d_params(in_channel,out_channel,kernel_size,do_bias),'conv2':conv2d_params(out_channel,out_channel,kernel_size,do_bias,init_scale=0.),'dense_t1':
        dense_params(t_emb_dim,out_channel),'skip':conv2d_params(in_channel,out_channel,1,do_bias)}
    def group_params(in_channel,
        out_channel,
        kernel_size=3,
        t_emb_dim=128):
        return{'group0':down_conv_params(in_channel*2,out_channel*4,kernel_size,t_emb_dim),
        'group1':down_conv_params(in_channel*4,out_channel*8,kernel_size,t_emb_dim),
        'group2':down_conv_params(in_channel*8,out_channel*8,kernel_size,t_emb_dim),
        'group3':down_conv_params(in_channel*8,out_channel*8,kernel_size,t_emb_dim),
        'group4':down_conv_params(in_channel*8,out_channel*8,kernel_size,t_emb_dim),
        'group5':down_conv_params(in_channel*8,out_channel*8,kernel_size,t_emb_dim)}

    p={'group_params':group_params(ngf,ngf,kernel_size,t_emb_dim)}
    t_f = TimestepEmbedding_ICFG(t_emb_dim,t_emb_dim,t_emb_dim,act=nn.LeakyReLU(0.2))
    p['t_emb']=t_f[1]
    t_emb_function = t_f[0]
    p['final_conv']= conv2d_params(ngf*8+1, ngf*8, kernel_size,do_bias)
    p['start_conv']= conv2d_params(nc,ngf*2,1,do_bias)
    p['end_linear']= dense_params(ngf*8,1)
    flat_params = utils.cast(utils.flatten(p))
    if requires_grad:
      utils.set_requires_grad_except_bn_(flat_params)
    def run_down_conv(input,params,t_emb,base,downsample):
        out = act_f(input)
        out = F.conv2d(out,params[base+'.conv1.w'],params[base+'.conv1.b'],padding=padding,stride=1)
        # print(out.shape)
        # print(t_emb.shape)
        # print(params[base+'.dense_t1.weight'].shape)
        # print(params[base+'.dense_t1.bias'].shape)
        out += F.linear(t_emb,params[base+'.dense_t1.weight'],params[base+'.dense_t1.bias'])[..., None, None]
        out = act_f(out)
        if downsample:
            out = up_or_down_sampling.downsample_2d(out, fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, fir_kernel, factor=2)
        out = F.conv2d(out,params[base+'.conv2.w'],params[base+'.conv2.b'],padding=padding,stride=1)
        skip = F.conv2d(input,params[base+'.skip.w'],params[base+'.skip.b'],padding=0,stride=1)
        out = (out + skip)/np.sqrt(2)
        return out
    def run_group(o,params,t_emb,downsample):
        for i in range(count):
            o = run_down_conv(o,params,t_emb,'group_params.group%d'%i,downsample=False if count ==0 else downsample)            
        return o
    def f(x,params,t_emb,x_t):
        # for k,v in sorted(params.items()):
        #     print('k name {}'.format(k))
        t_emb_result = act_f(t_emb_function(t_emb,params,'t_emb'))
        input_x = torch.cat((x,x_t),dim=1)
        h0 = F.conv2d(input_x,params['start_conv.w'],params['start_conv.b'],padding=0,stride=1)
        out = run_group(h0,params,t_emb_result,downsample)

        batch, channel, height, width = out.shape
            
        stddev_group = 4
        stddev_feat = 1
    
        group = min(batch, stddev_group)
        stddev = out.view(
            group, -1,stddev_feat, channel //stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
    
        out = F.conv2d(out,params['final_conv.w'],params['final_conv.b'],padding=1)
        out = act_f(out)
   
    
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = F.linear(out,params['end_linear.weight'],params['end_linear.bias'])
    
        
        return out

    return f,flat_params
