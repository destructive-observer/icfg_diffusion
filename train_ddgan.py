# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
from ast import arg
import torch
import numpy as np

import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.lmdb_datasets import LMDBDataset


from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from score_sde.models.discriminator import Discriminator_large_icfg
from train_icfg import cfggan
from torch.nn.init import normal_
from icfg.data import ImageFolder
from icfg.data import ImageFolder_ImageNet

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * \
        (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):

        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)
#     print(a.s_cum)
#     print(a.s_cum.shape)
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
        extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
        extract(coeff.sigmas, t+1, x_start.shape) * noise

    return x_t, x_t_plus_one
# %% posterior sampling


class Posterior_Coefficients():
    def __init__(self, args, device):

        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32,
                          device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * \
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):

    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(
            coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, params, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, params, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x

# %%


def train(rank, gpu, args):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large
    from score_sde.models.discriminator import Discriminator_small_icfg,Discriminator_small_icfg,Discriminator_mid_icfg,Discriminator_mid_icfg2
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp_ICFG
    from EMA import EMA

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    if args.dataset == 'cifar10':
        dataset = CIFAR10(args.data_root, train=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)

    elif args.dataset == 'stackmnist':
        train_transform, valid_transform = _data_transforms_stacked_mnist()
        dataset = StackedMNIST(
            root=args.data_root, train=True, download=True, transform=train_transform)
    elif args.dataset == 'Imagenet64':
        train_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
#         dataset = ImageFolder(args.data_root,transform=train_transform)
        dataset = ImageFolder_ImageNet(args.data_root,transform=train_transform)
        # train_data = LSUN(root=args.data_root, classes=[
        #                   'church_outdoor_train'], transform=train_transform)
        # subset = list(range(0, 120000))
        # dataset = torch.utils.data.Subset(train_data, subset)
    elif args.dataset == 'lsun64':

        train_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = LSUN(root=args.data_root, classes=[
                          'church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)
    elif args.dataset == 'lsun256':
    
        train_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = LSUN(root=args.data_root, classes=[
                          'church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)
    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = ImageFolder(args.data_root,transform=train_transform)
#         dataset = LMDBDataset(root=args.data_root, name='celeba',
#                               train=True, transform=train_transform)
    elif args.dataset == 'lsun_bedroom64':
    
        train_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = LSUN(root=args.data_root, classes=[
                          'bedroom_train'], transform=train_transform)
        subset = list(range(0, 1300000))
        dataset = torch.utils.data.Subset(train_data, subset)
    elif args.dataset == 'lsun_tower64':
    
        train_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = LSUN(root=args.data_root, classes=[
                          'tower_train'], transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              sampler=train_sampler,
                                              drop_last=True)
    # netG, netG_param = NCSNpp_ICFG(args)

    # netG_optim_params = [
    #     {'params': [v for k, v in sorted(netG_param.items()) if v.requires_grad]}]
    # netG = NCSNpp(args).to(device)

    # if args.dataset == 'cifar10' or args.dataset == 'stackmnist':
    #     # netD = Discriminator_small(nc = 2*args.num_channels, ngf = args.ngf,
    #     #                        t_emb_dim = args.t_emb_dim,
    #     #                        act=nn.LeakyReLU(0.2)).to(device)
    #     netD, netD_param = Discriminator_small_icfg(nc=2*args.num_channels, ngf=args.ngf,
    #                                                 t_emb_dim=args.t_emb_dim,
    #                                                 act=nn.LeakyReLU(0.2), downsample=True)
    #     # netD_param = netD_param.to(device)
    # else:
    #     netD = Discriminator_large_icfg(nc=2*args.num_channels, ngf=args.ngf,
    #                                t_emb_dim=args.t_emb_dim,
    #                                act=nn.LeakyReLU(0.2)).to(device)
    # netD_optim_params = [
    #     {'params': [v for k, v in sorted(netD_param.items()) if v.requires_grad]}]
    # print()
    # for k,v in sorted(netD_param.items()):
    #     print(v.requires_grad)
    # print(netG.parameters())
    # broadcast_params(netG.parameters())
    # for i,value in enumerate(netD_optim_params[0]['params']):
    #     dist.broadcast(value, src=0)
    # broasdcast_params(netD_optim_params)
    # broadcast_params(netG.parameters())
    # with open('para_new.txt','a') as f:
    #     for k,v in sorted(netD_param.items()):
    #         f.write('name_{}\n'.format(k))
    # for layer in netD.named_modules():
    #     f.write('name_{}\n'.format(layer))
    # for layer in netG.named_modules():
    #     f.write('name_{}\n'.format(layer))

    # optimizerD = optim.Adam(netD_optim_params, lr=args.lr_d, betas = (args.beta1, args.beta2))
    # optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    # for group in optimizerD.param_groups:
    #     print(group)

    # optimizerG = optim.Adam(netG_optim_params, lr=args.lr_g, betas = (args.beta1, args.beta2))
    # optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    # for group in optimizerG.param_groups:
    #     print(group)
    # if args.use_ema:
    #     optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    # schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    # schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    # ddp
    # netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    # netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    exp = args.exp
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models',
                            os.path.join(exp_path, 'score_sde/models'))

    # coeff = Diffusion_Coefficients(args, device)
    # pos_coeff = Posterior_Coefficients(args, device)
    # T = get_time_schedule(args, device)

    # if args.resume:
    #     checkpoint_file = os.path.join(exp_path, 'content.pth')
    #     checkpoint = torch.load(checkpoint_file, map_location=device)
    #     init_epoch = checkpoint['epoch']
    #     epoch = init_epoch
    #     netG.load_state_dict(checkpoint['netG_dict'])
    #     # load G

    #     optimizerG.load_state_dict(checkpoint['optimizerG'])
    #     schedulerG.load_state_dict(checkpoint['schedulerG'])
    #     # load D
    #     netD.load_state_dict(checkpoint['netD_dict'])
    #     optimizerD.load_state_dict(checkpoint['optimizerD'])
    #     schedulerD.load_state_dict(checkpoint['schedulerD'])
    #     global_step = checkpoint['global_step']
    #     print("=> loaded checkpoint (epoch {})"
    #               .format(checkpoint['epoch']))
    # else:
    #     global_step, epoch, init_epoch = 0, 0, 0
    from_file = None
    saved_begin = None
    if args.saved:
        from_file = torch.load(args.saved, map_location=None if torch.cuda.is_available() else 'cpu')
        saved = args.saved
        saved_begin = args.savedbegin
        new_interval = args.save_interval
        stages = args.num_stages
#       logging('WARNING: from file is begin')
        args = from_file['opt']
        args.save_interval = new_interval
        args.num_stages = stages
        
    def z_gen(num):
        return normal_(torch.Tensor(num, 100), std=1.0)

    if args.dataset == 'cifar10' or args.dataset == 'stackmnist':
        print('xxx')
        cfggan(args, Discriminator_small_icfg,
           NCSNpp_ICFG, z_gen, data_loader, device,from_file,saved_begin)
        
    elif args.dataset.endswith('64'):
        print('ggg')
        cfggan(args,Discriminator_mid_icfg2,
           NCSNpp_ICFG, z_gen, data_loader, device,from_file,saved_begin)
#         cfggan(args,Discriminator_large_icfg,
#            NCSNpp_ICFG, z_gen, data_loader, device)
        
    else:
        print('ccc')
        cfggan(args, Discriminator_large_icfg,
           NCSNpp_ICFG, z_gen, data_loader, device,from_file,saved_begin)
       

    # for epoch in range(init_epoch, args.num_epoch+1):
    #     train_sampler.set_epoch(epoch)

    #     for iteration, (x, y) in enumerate(data_loader):
    #         # for k,v in sorted(netD_param.items()):
    #         #     v.requires_grad = True
    #             # if v. grad is not None:
    #             #     v.grad.detach_()
    #             #     v.grad.zero_()
    #         # print('begin')
    #         for i,value in enumerate(netD_optim_params[0]['params']):
    #             value.requires_grad = True
    #         #     print('value.is_leaf {}'.format(value.is_leaf))
    #         #     print('value.requires_grad {}'.format(value.requires_grad))
    #         # # print('end')
    #         # for p in netD.parameters():
    #         #     p.requires_grad = True
    #         # for k,v in sorted(netD_param.items()):
    #         #     if v.grad is not None:
    #         #         print(v.grad)
    #         optimizerD.zero_grad()
    #         # for k,v in sorted(netD_param.items()):
    #         #     if v.grad is not None:
    #         #         print(v.requires_grad)
    #         # netD.zero_grad()

    #         #sample from p(x_0)
    #         real_data = x.to(device, non_blocking=True)

    #         #sample t
    #         t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

    #         x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
    #         x_t.requires_grad = True

    #         # train with real
    #         D_real = netD(x_t,netD_param, t, x_tp1.detach()).view(-1)

    # errD_real = F.softplus(-D_real)
    #         errD_real = errD_real.mean()

    #         errD_real.backward(retain_graph=True)

    #         if args.lazy_reg is None:
    #             grad_real = torch.autograd.grad(
    #                         outputs=D_real.sum(), inputs=x_t, create_graph=True
    #                         )[0]
    #             grad_penalty = (
    #                             grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    #                             ).mean()

    #             grad_penalty = args.r1_gamma / 2 * grad_penalty
    #             grad_penalty.backward()
    #         else:
    #             if global_step % args.lazy_reg == 0:
    #                 grad_real = torch.autograd.grad(
    #                         outputs=D_real.sum(), inputs=x_t, create_graph=True
    #                         )[0]
    #                 grad_penalty = (
    #                             grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    #                             ).mean()

    #                 grad_penalty = args.r1_gamma / 2 * grad_penalty
    #                 grad_penalty.backward()

    #         # train with fake
    #         latent_z = torch.randn(batch_size, nz, device=device)

    #         x_0_predict = netG(x_tp1.detach(),netG_param, t, latent_z)
    #         x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

    #         output = netD(x_pos_sample,netD_param, t, x_tp1.detach()).view(-1)

    #         errD_fake = F.softplus(output)
    #         errD_fake = errD_fake.mean()
    #         errD_fake.backward()

    #         errD = errD_real + errD_fake
    #         # for k,v in sorted(netD_param.items()):
    #         #     if v.grad is not None:
    #         #         print(v.grad)
    #         # Update D
    #         optimizerD.step()
    #         # for k,v in sorted(netD_param.items()):
    #         #     if v.grad is not None:
    #         #         print(v.grad)

    #         #update G
    #         # for p in netD.parameters():
    #         #     p.requires_grad = False
    #         for i,value in enumerate(netD_optim_params[0]['params']):
    #             value.requires_grad = False
    #         # for i,value in enumerate(netD_optim_params[0]['params']):
    #         #     value.requires_grad = False
    #         # netG.zero_grad()
    #         optimizerG.zero_grad()

    #         t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

    #         x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

    #         latent_z = torch.randn(batch_size, nz,device=device)

    #         x_0_predict = netG(x_tp1.detach(),netG_param, t, latent_z)
    #         x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

    #         output = netD(x_pos_sample,netD_param, t, x_tp1.detach()).view(-1)

    #         errG = F.softplus(-output)
    #         errG = errG.mean()

    #         errG.backward()
    #         optimizerG.step()

    #         global_step += 1
    #         if iteration % 100 == 0:
    #             if rank == 0:
    #                 print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch,iteration, errG.item(), errD.item()))

    #     if not args.no_lr_decay:

    #         schedulerG.step()
    #         schedulerD.step()

    #     if rank == 0:
    #         if epoch % 10 == 0:
    #             torchvision.utils.save_image(x_pos_sample, os.path.join(exp_path, 'xpos_epoch_{}.png'.format(epoch)), normalize=True)

    #         x_t_1 = torch.randn_like(real_data)
    #         fake_sample = sample_from_model(pos_coeff, netG,netG_param, args.num_timesteps, x_t_1, T, args)
            # torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)), normalize=True)

    #         if args.save_content:
    #             if epoch % args.save_content_every == 0:
    #                 print('Saving content.')
    #                 content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
    #                            'netG_dict': netG_param, 'optimizerG': optimizerG.state_dict(),
    #                            'schedulerG': schedulerG.state_dict(),
    #                         #    'netD_dict': netD.state_dict(),
    #                            'netD_dict': netD_param,
    #                            'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}

    #                 torch.save(content, os.path.join(exp_path, 'content.pth'))

    #         if epoch % args.save_ckpt_every == 0:
    #             if args.use_ema:
    #                 optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

    #             torch.save(netG_param, os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
    #             if args.use_ema:
    #                 optimizerG.swap_parameters_with_ema(store_params_in_ema=True)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(
        backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


def add_icfg_args_(parser):
    #---  proc
    #    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    #    parser.add_argument('--dataset', type=str, choices=[MNIST, SVHN, Bedroom64, Church64,CIFAR10,Tower64,Brlr64,Twbg64,FashionMNIST,EMNIST,Bedroom256,Bedroom128,CelebaHQ,Celeba128,Celeba256,ImageNet,ImageNet64], required=True, help='Dataset.')
    #    parser.add_argument('--dataroot', type=str, default='.')
    #    parser.add_argument('--model', type=str, choices=[DCGANx,Resnet4,FC2,Resnet3,DCGAN4,Balance,Resnet256,Resnet128,Resnet1024,Balance2], help='Model.')
    #    parser.add_argument('--norm_type', type=str, default='bn', choices=['bn','none'], help="'bn': batch normalization, 'none': no normalization")
    #    parser.add_argument('--batch_size', type=int, default=64, help='Number of images for training.')
    #    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for retrieving images.')

    #---  cfggan
    parser.add_argument('--cfg_T', type=int, default=1, help='T for ICFG.')
    parser.add_argument('--cfg_U', type=int, default=1,
                        help='U (discriminator update frequency) for ICFG.')
    parser.add_argument('--cfg_N', type=int, default=640,
                        help='N (number of generated examples used for approximator training).')
    parser.add_argument('--num_stages', type=int,
                        default=10000, help='Number of stages.')
    parser.add_argument('--cfg_eta', type=float, default=1,
                        help='Generator step-size eta.')
    parser.add_argument('--divergence', type=str, default='KL',
                        help='divergence type KL, logd, JS, Jeffrey.')
    parser.add_argument('--gftype', type=str, default='kl',
                        help='gradient flow type. kl or ws-kl, ws-js, ws-logd. kl is cfg')
    parser.add_argument('--noise_factor', type=float, default=1.e-2,
                        help='noise_factor for ws gradient flow. other type gradient flow is 0')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='Learning rate used for training discriminator and approximator.')
    parser.add_argument('--cfg_x_epo', type=int, default=10,
                        help='Number of epochs for approximator training.')

    parser.add_argument('--gen', type=str,
                        help='Pathname for saving generated images.')
    parser.add_argument('--save', type=str, default='',
                        help='Pathname for saving models.')
    parser.add_argument('--save_interval', type=int, default=-1,
                        help='Interval for saving models. -1: no saving.')
    parser.add_argument('--gen_interval', type=int, default=50,
                        help='Interval for generating images. -1: no generation.')
    parser.add_argument('--num_gen', type=int, default=10,
                        help='Number of images to be generated.')
    parser.add_argument('--gen_nrow', type=int, default=5,
                        help='Number of images in each row when making a collage of generated of images.')
    parser.add_argument('--diff_max', type=float, default=40,
                        help='Stop training if |D(real)-D(gen)| exceeds this after passing the initial starting-up phase.')
    parser.add_argument('--lamda', type=float, default='0',
                        help='0 stands for no constraint or others stands for lamda value')
    parser.add_argument('--scale', type=float, default='0.1',
                        help='scale for the eposion, value is [0,1]')
    parser.add_argument('--gptype', type=int, default='0',
                        help='0-0 centered 1-1 centered 2-newtype.')
    parser.add_argument('--app_type', type=int, default='0',
                        help='0-icfg real img 1-new iter real img 2-no real img.')
    parser.add_argument('--alpha', type=float, default='1',
                        help='use w distance to regulation regression')
    parser.add_argument('--saved', type=str, default='', help='Pathname for the saved model.')
    parser.add_argument('--savedbegin', type=int, default=0, help='begin stage for the saved model.')
    parser.add_argument('--verbose', action='store_true',
                        help='If true, display more info.')
    return parser


def add_sep_for_dict(split='-', **dict):
    str_list=''
    for k, v in dict.items():
        str_list += k+str(v)+split
    return str_list


#    parser.add_argument('--verbose', action='store_true', help='If true, display more info.')
#    parser.add_argument('--saved', type=str, default='', help='Pathname for the saved model.')
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    #geenrator and training
    parser.add_argument(
        '--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--data_root', default='./', help='dir of dataset')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int,
                        default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float,
                        default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float,
                        default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float,
                        default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=50,
                        help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int,
                        default=25, help='save ckpt every x epochs')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6020',
                        help='port for master address')

    # --------------------add icfg parameters
    parser = add_icfg_args_(parser)

    args = parser.parse_args()
    # --icfg optimal
    args.approx_redmax = 5
    args.approx_decay = 0.1
    args.weight_decay = 0.0
    args.cfg_N = args.batch_size * args.cfg_T
#     args.cfg_N = args.batch_size
    RMSprop_str = 'RMSprop'
    Adam_str = 'Adam'
    args.optim_type = Adam_str
    args.optim_eps = 1e-18
    args.optim_a1 = 0.5
    args.optim_a2 = 0.9
    # ----------
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    gen_dir_dict = {}
    gen_dir = None
    if args.gen is None and args.num_gen > 0 and args.gen_interval > 0:
        # gen_dir_dict['gen'] = ''
        gen_dir_dict['dataset'] = args.dataset
        gen_dir_dict['cfg_eta'] = args.cfg_eta
        gen_dir_dict['lr_g'] = args.lr_g
        gen_dir_dict['lr_d'] = args.lr_d
        gen_dir_dict['alpha'] = args.alpha
        gen_dir_dict['cfg_T'] = args.cfg_T
        gen_dir_dict['cfg_N'] = args.cfg_N
        gen_dir_dict['num_timesteps'] = args.num_timesteps
        gen_dir_dict['num_stages'] = args.num_stages
        gen_dir_dict['app_type'] = 'app_type'+str(args.app_type)
        if args.lamda != 0:
            gen_dir_dict['gptype'] = args.gptype
            gen_dir_dict['lamda'] = args.lamda
            gen_dir_dict['scale'] = args.scale
        # if args.model is not None:
        #     gen_dir_dict['model'] = args.model
        gen_dir = add_sep_for_dict('_', **gen_dir_dict)
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        print(gen_dir)
        args.gen = gen_dir + os.path.sep+args.dataset
    if args.save =='':
        save = 'mod'
        if not os.path.exists(save):
            os.makedirs(save)
        gen_dir = add_sep_for_dict('-', **gen_dir_dict)
        print(gen_dir)
        save = save+os.path.sep+gen_dir
        args.save = save

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' %
                  (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(
                global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')

        init_processes(0, size, train, args)
