import os,time
from tkinter import N
import torch
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader
import torchnet as tnt
from torch.optim import RMSprop, Adam, SGD
from torchvision.utils import save_image

from icfg.utils.utils import cast
from icfg.utils.utils0 import timeLog, copy_params, clone_params, print_params, print_num_params, stem_name
from icfg.utils.utils0 import raise_if_absent, add_if_absent_, logging, raise_if_nonpositive_any, raise_if_nan
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
# from logger import Logger
import math
import copy
from score_sde.models.dense_layer import variance_scaling_init_
White=255
RMSprop_str='RMSprop'
Adam_str='Adam'
from EMA import EMA
# vizG = visdom.Visdom(env='G2')  # 初始化visdom类
# vizD = visdom.Visdom(env='D2')
# vizG_n = visdom.Visdom(env='Gn')
# vizG_f = visdom.Visdom(env='fn')

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
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
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
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
#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator,params, n_time, x_init, T, z):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            # latent_z = torch.randn(x.size(0), nz, device=x.device)
            x_0 = generator(x, params,t_time, z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        
    return x
 
def sample_from_model_icfg(coefficients, n_time, x_init,ddg,opt):
    x = x_init
    for i in reversed(range(n_time)):
      # t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
      # print(i)
      x_0 = ddg.generate(opt.num_gen,x_tp1_gen=x,list_or_not=False,t_time=i+1)[0]
      # x_0 = x_0.cuda()
      # x_new = sample_posterior(coefficients, x_0, x, t)
      x = x_0.detach()
        
    return x
 
def d_loss_dflt(d_out_real, d_out_fake, alpha):
   return (  torch.log(1+torch.exp((-1)*d_out_real)) 
           + torch.log(1+torch.exp(     d_out_fake)) ).mean()
def g_loss_dflt(fake, target_fake):
   num = fake.size(0)
   r1 = ((fake - target_fake)**2).sum()/2/num
   # r2 = 0.1*torch.mean(target_fake)
   return r1
def d_loss_wgan(d_out_real,d_out_fake, alpha):
   d_logistic_loss=(torch.log(1+torch.exp((-1)*d_out_real)) 
           + torch.log(1+torch.exp(     d_out_fake)) ).mean()
   w_loss = ((-1)*torch.mean(d_out_fake) + torch.mean(d_out_real))
   # print(d_out_real)
   # print(d_out_fake)
   # print(d_logistic_loss)
   # print(d_logistic_loss.shape)
   # alpha = torch.cuda.FloatTensor(np.random.random(1))
   # alpha1 = alpha
   loss1 = alpha*d_logistic_loss
   # loss1 = 0
   loss2 = (1-alpha)*w_loss
   # wgan_loss = d_logistic_loss + ((1-alpha)*w_loss)
   # wgan_loss = ((-1)*torch.mean(d_out_fake) + torch.mean(d_out_real))/torch.mean(d_out_real)
   # wgan_loss = torch.exp((-1)*torch.mean(d_out_real) + torch.mean(d_out_fake)/torch.mean(d_out_real))
   # gp = wgan_gp(self,fake,real,LAMBDA,netD)
   return loss1,loss2
def wgan_gp(self,fake,real,LAMBDA,netD,centered,t_emb,x_tp1):
   real_data = real
   real_data = real_data.cuda()
   fake_data = fake
   fake_data=fake_data.cuda()
   # netD = self.D
            # alpha = torch.rand(real.size(0),1,1, 1)
            # alpha = alpha.expand(real_data.size())
   alpha = torch.cuda.FloatTensor(np.random.random((real_data.size(0),1,1,1)))
            # alpha = alpha.cuda()
            # alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha
            # print('real_data shape is {}'.format(real_data.shape))
            # print('fake_data shape is {}'.format(fake_data.shape))
   interpolates = alpha * real_data + ((1 - alpha) * fake_data)
            # print('interpolates shape is {}'.format(interpolates.shape))
            # interpolates = interpolates.to(device)#.cuda()
   interpolates = interpolates.cuda()
   interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
   disc_interpolates = netD(interpolates,t_emb,x_tp1.detach())
   gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),#.cuda(), #if use_cuda else torch.ones(
                                    #disc_interpolates.size()),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]#LAMBDA = 1
            # print('gradients.size {}'.format(gradients.shape)
   gradients = gradients.reshape(gradients.size(0),-1)
   gradient_penalty = ((gradients.norm(2, dim=1) - centered) ** 2).mean() * LAMBDA
        
   return gradient_penalty
def new_gp(self,fake,real,LAMBDA,netD,centered,t_emb,x_tp1,scale=0.1):
   real_data = real
   real_data = real_data.cuda()
   fake_data = fake
   fake_data=fake_data.cuda()
   # netD = self.D
            # alpha = torch.rand(real.size(0),1,1, 1)
            # alpha = alpha.expand(real_data.size())
   alpha = torch.cuda.FloatTensor(np.random.random((real_data.size(0),1,1,1)))
            # alpha = alpha.cuda()
            # alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha
            # print('real_data shape is {}'.format(real_data.shape))
            # print('fake_data shape is {}'.format(fake_data.shape))
   interpolates = alpha * real_data + ((1 - alpha) * fake_data)
            # print('interpolates shape is {}'.format(interpolates.shape))
            # interpolates = interpolates.to(device)#.cuda()
   interpolates = interpolates.cuda()
   interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
   disc_interpolates = netD(interpolates,t_emb,x_tp1.detach())
   gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),#.cuda(), #if use_cuda else torch.ones(
                                    #disc_interpolates.size()),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]#LAMBDA = 1
            # print('gradients.size {}'.format(gradients.shape)
   gradients = gradients.reshape(gradients.size(0),-1)
   # print('gradient before {}'.format(((gradients.norm(2, dim=1) - centered) ** 2).mean()))
   eposion = 1/(gradients.shape[1])#C*N^2
   # print(real_data.shape[1])
   # print(real_data.shape[2])
   # print(eposion)
   # gradients1 = gradients-0.00000125
   gradients1 = gradients-math.sqrt(eposion*scale)
   # print('gradients1 {}'.format(math.sqrt(eposion*scale)))
   # gradients1 = gradients-0.05
   # print(1/math.sqrt(gradients.shape[1]))
   # as for convinent, the k is to set 1/imagesz * 1.732 * (0.1) n = imagesz *imagesz*3
   # so the sqrt(n) = imagesz * sqrt(3) k = 1/sqrt(n)*(s) s is about 0.1
   # for imagesez = 32, 64, ..., 1024
   # the k is about 0.009 to 0.00056
   #
   # gradients1 = gradients-(1/math.sqrt(gradients.shape[1]))*(0.1)
   gradient_penalty = ((gradients1.norm(2, dim=1) - centered) ** 2).mean() * LAMBDA
   # print('gradient_penalty {}'.format(gradient_penalty))
   # print(interpolates.shape)
   # print(interpolates.mean())
   # interpolates1 = interpolates.reshape(interpolates.size(0),-1)
   # print(interpolates1.shape)
   # print(disc_interpolates.shape)
   # print(interpolates1.mean(keepdim=True).shape)

   # gradient_penalty =  ((disc_interpolates - interpolates).norm(2, dim=1)).mean()*LAMBDA
   
   # print(gradient_penalty.shape)
   return gradient_penalty
#-----------------------------------------------------------------
def is_last(opt, stage):
   return stage == opt.num_stages-1
def is_time_to_save(opt, stage):
   return opt.save_interval > 0 and (stage+1)%opt.save_interval == 0 or is_last(opt, stage)
def is_time_to_generate(opt, stage):
   return opt.gen_interval > 0 and (stage+1)%opt.gen_interval == 0 or is_last(opt, stage)
      
#-----------------------------------------------------------------
def cfggan(opt, d_config, g_config, z_gen, loader,fromfile=None,saved=None,
           d_loss=d_loss_wgan, g_loss=g_loss_dflt):

#    check_opt_(opt)
 
   write_real(opt, loader)
   # write_real_num(opt, loader)
   

   optim_config = OptimConfig(opt)
   begin = 0
   ddg = DDG(opt, d_config, g_config, z_gen, optim_config,fromfile)
#    timeLog('saved ' + str(saved) + '.')
   # if fromfile != None:
   #    ddg.initialize_G(g_loss, opt.cfg_N)
   iterator = None
   torch.autograd.set_detect_anomaly(True)
   if saved:
      begin = saved[-8:-4]
      timeLog('stages ' + str(begin) + '.')
      begin = int(begin)
   else:
      ddg.initialize_G(g_loss, opt.cfg_N,loader,iterator)

   #---  xICFG
   
   for stage in range(begin,opt.num_stages):   
      timeLog('xICFG stage %d -----------------' % (stage+1))
      # if stage == 0:
      #    cfg_eta = opt.cfg_eta
      #    ddg.cfg_eta = 10
      # else:
      #    ddg.cfg_eta = cfg_eta
      # timeLog('cfg_eta %d -----------------' % (ddg.cfg_eta))
      real_datas,diff,d_loss_v,d_loss_gp,d_fake,d_real = ddg.icfg(loader, iterator, d_loss, opt.cfg_U)
      ddg.epoch = stage
      # print('ddg.epoch {}'.format(ddg.epoch))
      # if stage >= 2000:
      #    change_lr_(ddg.d_optimizer,optim_config.lr0)
      # if stage >= 5000:
      #    change_lr_(ddg.d_optimizer,optim_config.lr0*0.1)
     
      if opt.diff_max > 0 and abs(diff) > opt.diff_max and stage >= 2000:
         timeLog('Stopping as |D(real)-D(gen)| exceeded ' + str(opt.diff_max) + '.')
         break

      if is_time_to_save(opt, stage):
         # swap_ema(opt, ddg, stage)
         save_ddg(opt, ddg, stage)
         # swap_ema(opt, ddg, stage)
      if is_time_to_generate(opt, stage):
         generate(opt, ddg, stage)        
         
      ddg.approximate(g_loss, opt.cfg_N,real_datas)         
      # ddg.tensorboard(stage, 'train',g_loss_v,d_loss_v,d_loss_gp,d_fake,d_real)
#-----------------------------------------------------------------
def write_real(opt, loader):
   timeLog('write_real: ... ')
   dir = 'real'
   if not os.path.exists(dir):
      os.mkdir(dir)

   real,_ = get_next(loader, None)
   real = real[0]   
   num = min(10, real.size(0))
   nm = dir + os.path.sep + opt.dataset + '-%dc'%num
   write_image(real[0:num], nm + '.jpg', nrow=2)
   # my_data = (real[0:num]+1)/2
   # vizD.images(my_data,opts=dict(title='real images write real', caption=' write real'))

#real image used for compute fid and is
def write_real_num(opt, loader,num=800):
   ## appoint the num = 1000
   timeLog('write_real_num: ... ')
   dir = 'real'
   if not os.path.exists(dir):
      os.mkdir(dir)

   real,_ = get_next(loader, None)
   real = real[0]
   index = 0
   total_num=0
   for num in range(num):
      while(index<real.size(0)):
         index+=1
         total_num+=1
         nm = dir + os.path.sep + opt.dataset + '-%dc'%total_num
         write_image(real[index-1:index], nm + 'xx.jpg', nrow=1)
      real,_ = get_next(loader, None)
      real = real[0]
      index = 0   
   # num = min(10, real.size(0))
   # for num in range(num):
   #    while(index<real.size(0)):
   #       index+=1  
   #       if index % 10 == 0:
   #          nm = dir + os.path.sep + opt.dataset + '-%dc'%num
   #          write_image(real[index-10:index], nm + 'x.jpg', nrow=5)
   #          continue
   #    real,_ = get_next(loader, None)
   #    real = real[0]
   #    index = 0


#-----------------------------------------------------------------
#  To make an inifinite loop over training data
#-----------------------------------------------------------------
def get_next(loader, iterator):
   # for i,data in enumerate(loader):
      # print('i{}and data{}'.format(i,data.shape)) 
   if iterator is None:
      iterator = iter(loader)   
   try:
      data = next(iterator)
   except StopIteration:
      logging('get_next: ... getting to the end of data ... starting over ...')
      iterator = iter(loader)
      data = next(iterator)
   return data,iterator

#-----------------------------------------------------------------
# DDG stands for D's (discriminators) and G (generator).  
#-----------------------------------------------------------------
class DDG:
   def __init__(self, opt, d_config, g_config, z_gen, optim_config, from_file=None):
      assert opt.cfg_T > 0
      self.verbose = opt.verbose
      # self.d_params_list = [d_config(nc = 2*opt.num_channels, ngf = opt.ngf, 
      #                              t_emb_dim = opt.t_emb_dim,
      #                              act=nn.LeakyReLU(0.2),downsample=True,requires_grad=False)[1] for i in range(opt.cfg_T) ]
      # self.d_net,self.d_params = d_config(nc = 2*opt.num_channels, ngf = opt.ngf, 
      #                              t_emb_dim = opt.t_emb_dim,
      #                              act=nn.LeakyReLU(0.2),downsample=True,requires_grad=True)
      # self.g_net,self.g_params = g_config(opt)
      self.z_gen = z_gen
      self.cfg_eta = opt.cfg_eta
      self.alpha = opt.alpha
      self.optim_config = optim_config
      self.d_optimizer = None
      self.g_optimizer = None
      self.current_time = time.strftime('%Y-%m-%d %H%M%S')
    #   self.logstr = opt.logstr
      self.lamda = opt.lamda
      self.gptype = opt.gptype
      self.scale = opt.scale
      self.epoch = 0
      self.device = opt.device
      print(self.device)
      self.coeff = Diffusion_Coefficients(opt, opt.device)
      self.pos_coeff = Posterior_Coefficients(opt, opt.device)
      self.time_step=opt.num_timesteps
      self.nz = opt.nz
      self.cfg_N = opt.cfg_N
      self.d_params_list=[]   
      self.rank = opt.local_rank
      self.world_size = opt.world_size
      self.device = opt.device
      self.gpu = opt.gpu
      self.batch_size = opt.batch_size
      
      
      for i in range(opt.cfg_T):
         a = d_config(nc = 2*opt.num_channels, ngf = opt.ngf, 
                                   t_emb_dim = opt.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(self.device)
         # a.train()
         self.d_params_list.append(a)
      self.d_net = d_config(nc = 2*opt.num_channels, ngf = opt.ngf, 
                                   t_emb_dim = opt.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(self.device)
      self.d_net = nn.SyncBatchNorm.convert_sync_batchnorm(self.d_net)
      self.d_net = nn.parallel.DistributedDataParallel(self.d_net, device_ids=[self.gpu],broadcast_buffers=False,find_unused_parameters = True)
      # self.d_net= self.d_net.cuda()
      # self.d_net.train()
      # self.d_params = self.d_net.named_parameters()
      self.g_net = g_config(opt).to(self.device)
      # self.g_net = self.g_net.cuda()
      self.g_net = nn.SyncBatchNorm.convert_sync_batchnorm(self.g_net)
      self.g_net = nn.parallel.DistributedDataParallel(self.g_net, device_ids=[self.gpu],broadcast_buffers=False,find_unused_parameters = False)
      # set_requirsgrad(self.g_net)
      # self.logger = Logger('./logs/' + str(opt.gen)+'-' +str(opt.cfg_eta)+ "/")
      if optim_config is not None:
         self.d_optimizer = optim_config.create_optimizer(self.d_net.parameters(),use_ema=False,type='d')

      if from_file is not None:
         self.load(from_file)

      logging('----  D  ----')
      # if self.verbose:         
      #    print_params(self.d_params)       
      # print_num_params(self.d_params) 
      
      # logging('----  G  ----')
      # if self.verbose:
      #    print_params(self.g_params)
      # print_num_params(self.g_params)          

   def check_trainability(self):
      if self.optim_config is None:
         raise Exception('This DDG is not trainalbe.')
   # def tensorboard(self, it, phase,g_loss,d_loss,d_loss_gp,d_fake,d_real):
   #      # (1) Log the scalar values
   #      prefix = phase+'/'
   #      info = {prefix + 'G_loss': g_loss,
   #             # prefix + 'G_adv_loss': self.g_adv_loss,
   #             #  prefix + 'G_add_loss': self.g_add_loss,
   #              prefix + 'D_loss': d_loss,
   #              prefix + 'D_gp_loss': d_loss_gp,
   #             #  prefix + 'D_add_loss': self.d_add_loss,
   #              prefix + 'D_adv_loss_fake': self._get_data(d_fake),
   #              prefix + 'D_adv_loss_real': self._get_data(d_real)}
   #    #   print('tensorboard lt is{}'.format(it))
   #      for tag, value in info.items():
   #          self.logger.scalar_summary(tag, value, it)
   def _get_data(self, d):
        return d.data.item() if isinstance(d, Variable) else d
   def save(self, opt, path):
      timeLog('Saving: ' + path + ' ... ')
      torch.save(dict(d_params_list=self.d_params_list,
                      d_params=self.d_net,
                      g_params=self.g_net,
                      d_optimizer = self.d_optimizer,
                     #  g_optimizer = self.g_optimizer,
                      cfg_eta=self.cfg_eta,
                      opt=opt), 
                 path)
                 
   def load(self,  d):
      assert len(self.d_params_list) == len(d['d_params_list'])
      for i in range(len(self.d_params_list)):
#          self.d_params_list[i] = copy.deepcopy(d['d_params_list'][i])
         self.d_params_list[i].load_state_dict(d['d_params_list'][i].state_dict())
         # copy_params(src=d['d_params_list'][i], dst=self.d_params_list[i])
#       self.d_net = copy.deepcopy(d['d_params'])
      self.d_net.load_state_dict(d['d_params'].state_dict())
#       self.g_net = copy.deepcopy(d['g_params'])
      self.g_net.load_state_dict(d['g_params'].state_dict())
      self.d_optimizer.load_state_dict(d['d_optimizer'].state_dict())
      # copy_params(src=d['d_params'], dst=self.d_params)
      # copy_params(src=d['g_params'], dst=self.g_params)      
      self.cfg_eta = d['cfg_eta']
   
   #----------------------------------------------------------
   def num_D(self):
      return len(self.d_params_list)
      
   def check_t(self, t, who):      
      if t < 0 or t >= self.num_D():
         raise ValueError('%s: t is out of range: t=%d, num_D=%d.' % (who,t,self.num_D()))
      
   def get_d_params(self, t):
      self.check_t(t, 'get_d_params')
      return self.d_params_list[t]
      
   def store_d_params(self, t):
      self.check_t(t, 'store_d_params')
      # copy_params(src=self.d_net, dst=self.d_params_list[t])
      self.d_params_list[t] = copy.deepcopy(self.d_net)
   #----------------------------------------------------------      
   def generate(self, num_gen, t=-1, do_return_z=False, batch_size=-1,real_data=None,list_or_not=None,t_time = None,x_tp1_gen=None):
      assert num_gen > 0
      if t < 0:
         t = self.num_D()
      if batch_size <= 0:
         batch_size = num_gen
         
      num_gened = 0
      fakes = None
      zs = None
      is_train = False
      t_emb_s = None
      x_tp1_s = None
      x_tp_s = None
      real_datas = None
      x_tp1 = None
      x_t = None
      while num_gened < num_gen:
         num = min(batch_size, num_gen - num_gened)
         with torch.no_grad():       
            z = self.z_gen(num)
            z = z.to(self.device, non_blocking=True)
            if t_time is not None:
               t_emb = torch.randint(t_time-1, t_time, (num,),device=self.device)
               # print(t_emb)      
            else:
               t_emb = torch.randint(0, self.time_step, (num,),device=self.device)       
            if x_tp1_gen is None:
               if list_or_not:
                  real_datas = real_data[num_gened:num_gened+num].to(self.device)
               else:
                  real_datas = real_data.to(self.device)
               x_t, x_tp1 = q_sample_pairs(self.coeff, real_datas, t_emb) 
            else:   
               x_tp1 =   x_tp1_gen.to(self.device)
               x_t = x_tp1_gen.to(self.device)
            
            fake = self.g_net(x_tp1.detach(), t_emb, z)
            fake = sample_posterior(self.pos_coeff, fake, x_tp1, t_emb)
            # x_t_1 = torch.randn_like(real_data,device=self.device)
            # fake = 
            # fake = sample_from_model(self.pos_coeff, self.g_net, self.g_params, self.time_step, x_t_1, None, z)
            ## must a sample from g_net. not sample from model.

            # print(fake)
            # fake = self.g_net(cast(z), self.g_params, is_train)
        #  if self.verbose:   
        #     if self.epoch % 50 == 0:
        #        vizG_n.images(fake,opts=dict(title='-1 - 1 fake images vizG_n+stage{}+numD xxxx'.format(self.epoch), caption='vizG_n D.'))   
        #        fake_g = (fake+1)/2
        #        vizG_n.images(fake_g,opts=dict(title='0 - 1 fake images vizG_n+stage{}+numD xxxx'.format(self.epoch), caption='vizG_n D.'))    

         for t0 in range(t):
            fake = fake.detach()
            if fake.grad is not None:
               fake.grad.zero_()
            fake.requires_grad = True
            # real_data=self.real_list[t0]
            # x_t, x_tp1 = q_sample_pairs(self.coeff, real_data, t_emb)
            # fake = sample_posterior(self.pos_coeff, x_0_predict, x_tp1, t)
            d_out = self.d_params_list[t0](fake,t_emb, x_tp1.detach()).view(-1)
            # d_out = self.d_net(fake, self.get_d_params(t0), True)
            # d_out_1 = d_out.view(-1,d_out.size(0))
            # if self.verbose:
            #    timeLog('DDG::generate ... with d_out={}'.format(d_out_1))
            # d_out = self.d_net(fake, self.d_params, True)
            d_out.backward(torch.ones_like(d_out))  
            # print(fake.grad.data) 
            fake.data += self.cfg_eta * fake.grad.data
            # vizG_n.images(fake.grad.data,opts=dict(title='-1 - 1 fake  grad images vizG_n+stage{}+numD {}'.format(self.epoch,t0), caption='vizG_n D.'))   
            # fake_g = (fake.grad.data+1)/2
            # vizG_n.images(fake_g,opts=dict(title='0 - 1 fake grad images vizG_n+stage{}+numD {}'.format(self.epoch,t0), caption='vizG_n D.'))    
#             print('fake data {}'.format(torch.mean(fake.data)))
#             print('fake data {}'.format(torch.mean(fake.grad.data)))
            if self.verbose:
               timeLog('DDG::generate ... with fake.data=%f and fake.grad=%f' % (torch.sum(fake.data),torch.sum(fake.grad.data)))
            # fake.data += self.cfg_eta * (fake.grad.data*torch.mean(self.real_sample)-torch.mean(d_out))/torch.mean(self.real_sample)/torch.mean(self.real_sample)
        #  if self.verbose:
        #     if self.epoch % 50 == 0:
        #        vizG_n.images(fake,opts=dict(title='-1 - 1 fake images vizG_n+stage{}+numD full'.format(self.epoch), caption='vizG_n D.'))   
        #        fake_g = (fake+1)/2
        #        vizG_n.images(fake_g,opts=dict(title='0 - 1 fake images vizG_n+stage{}+numD full'.format(self.epoch), caption='vizG_n D.')) 
         if fakes is None:
            sz = [num_gen] + list(fake.size())[1:]
            # print(sz)
            fakes = torch.Tensor(torch.Size(sz), device=torch.device('cpu'))
         if t_emb_s is None:
            st = [num_gen]
            t_emb_s = torch.LongTensor(torch.Size(st), device=torch.device('cpu'))
         if x_tp1_s is None:
            t_s = [num_gen]+ list(x_tp1.size())[1:]
            x_tp1_s = torch.Tensor(torch.Size(t_s), device=torch.device('cpu'))
         if x_tp_s is None:
            tp_s = [num_gen]+ list(x_t.size())[1:]
            x_tp_s = torch.Tensor(torch.Size(tp_s), device=torch.device('cpu'))
            # print(fakes)
         # print(fakes)
         # print(fake)
         # fake = sample_posterior(self.pos_coeff, fake, x_tp1, t_emb)
         fakes[num_gened:num_gened+num] = fake.to(torch.device('cpu'))
         t_emb_s [num_gened:num_gened+num] = t_emb
         x_tp1_s[num_gened:num_gened+num] = x_tp1.to(torch.device('cpu'))
         x_tp_s[num_gened:num_gened+num] = x_t.to(torch.device('cpu'))

         if do_return_z:
            if zs is None:  
               sz = [num_gen] + list(z.size())[1:]            
               zs = torch.Tensor(torch.Size(sz), device=torch.device('cpu'))
            zs[num_gened:num_gened+num] = z
           
 
         num_gened += num

      fakes.detach_()
      if do_return_z:
         return fakes, zs,t_emb_s,x_tp_s,x_tp1_s
      else:
         return fakes,t_emb_s,x_tp_s,x_tp1_s

   #-----------------------------------------------------------------
   def icfg(self, loader, iter, d_loss, cfg_U):   
      if self.rank == 0:
        timeLog('DDG::icfg ... ICFG with cfg_U=%d' % cfg_U)
      #   timeLog('DDG::icfg ... ICFG with settings=%s' % self.logstr)
      self.check_trainability()
      t_inc = 1 if self.verbose else 5
      is_train = True
      real_datas=None
      cfg_N=self.cfg_N
      num_gened=0  
      for t in range(self.num_D()):
         # print('self.num_D {}'.format(self.num_D()))
         sum_real = sum_fake = count = 0
         print('t{}'.format(t))
         for upd in range(cfg_U):
            sample,iter = get_next(loader, iter)

            num = sample[0].size(0)
            if real_datas is None:
               sz = [cfg_N*num] + list(sample[0].size())[1:]
               real_datas = torch.Tensor(torch.Size(sz), device=torch.device('cpu')) 
            # print('num is {}'.format(sample[0]))
            real_datas[num_gened:num_gened+num] = sample[0].to(torch.device('cpu'))
            num_gened += num
            
            sample[0]= sample[0].to(self.device, non_blocking=True)
#             self.real_sample = sample[0]
            # self.store_real_list(t,sample[0])
            fake,t_emb,x_t,x_tp1 = self.generate(num, t=t,real_data=sample[0],list_or_not=False)
            ####### the t_emb must Corresponding  to fake #####
            # t_emb = torch.randint(0, self.time_step, (sample[0].size(0),),device=self.device)
            fake = fake.to(self.device, non_blocking=True)
            t_emb = t_emb.to(self.device, non_blocking=True)
            x_t = x_t.to(self.device, non_blocking=True)
            x_tp1 = x_tp1.to(self.device, non_blocking=True)
#             print(fake)
#             print(x_t)
#             print(x_tp1)
            d_out_real = self.d_net(x_t,t_emb, x_tp1.detach()).view(-1)
            d_out_fake = self.d_net(fake,t_emb, x_tp1.detach()).view(-1)
#             print(d_out_real)
#             print(d_out_fake)
            loss1,loss2 = d_loss(d_out_real, d_out_fake,self.alpha)
            loss = loss1 + loss2
#             loss.backward()
            loss_gp=0
            # print(self.lamda)
            if self.lamda != 0:
               # timeLog('DDG::icfg ... ICFG with lamda=%s' % str(self.lamda))
               # loss_gp = wgan_gp(self,fake,sample[0],self.lamda,self.d_net,self.d_params)
               # print(self.lamda)
               if self.gptype ==0:
                  # print('0 ----{}'.format(self.gptype))
                  loss_gp = wgan_gp(self,fake,sample[0],self.lamda,self.d_net,0,t_emb,x_tp1.detach())
#                   loss_gp.backward()
               elif self.gptype ==1:
                  # print('1 ----{}'.format(self.gptype))
                  loss_gp = wgan_gp(self,fake,sample[0],self.lamda,self.d_net,1,t_emb,x_tp1.detach())
#                   loss_gp.backward()
               elif self.gptype ==2:
                  # print('2 ----{}'.format(self.gptype))
                  loss_gp = new_gp(self,fake,sample[0],self.lamda,self.d_net,0,t_emb,x_tp1.detach(),scale=self.scale)
#                   loss_gp.backward()
               else:
                  raise ValueError('Unknown gptype: %s ...' % self.gptype)
#             print(loss)
#             print(loss_gp)
            loss = loss + loss_gp
            print(loss)
            loss.backward()
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()         
            
            with torch.no_grad():
               sum_real += float(d_out_real.sum()); sum_fake += float(d_out_fake.sum()); count += num            
            
         self.store_d_params(t)
         
         if t_inc > 0 and ((t+1) % t_inc == 0 or t == self.num_D()-1) and self.rank == 0:
            logging('  t=%d: real,%s, fake,%s ' % (t+1, sum_real/count, sum_fake/count))
            logging('  t=%d: loss_logistic,%s, loss_wgan,%s ' % (t+1, loss1, loss2))

      raise_if_nan(sum_real)
      raise_if_nan(sum_fake)

      return real_datas,(sum_real-sum_fake)/count,loss,loss_gp,sum_fake/count,sum_real/count

   #-----------------------------------------------------------------
   def initialize_G(self, g_loss, cfg_N,loader,iterator): 
      timeLog('DDG::initialize_G ... Initializing tilde(G) ... ')
      z = self.z_gen(1)
      z = z.to(self.device, non_blocking=True)
      sample,iterator = get_next(loader, iterator)
      real_data = sample[0].to(self.device)
      # x_t_1 = torch.randn_like(sample[0],device=self.device)
      # g_out = sample_from_model(self.pos_coeff, self.g_net, self.g_params, self.time_step, x_t_1, None, z)

      # g_out = self.g_net(cast(z), self.g_params, False)
      ## sample batch_size = 1
      t_emb = torch.randint(0, self.time_step, (1,),device=self.device)
      # # print(sample[0][0].shape)
      # # print(t_emb.shape)
      # # print(self.coeff.device)
      # print(real_data[0].device)
      # print(t_emb.device)
      # print(self.device)
      x_t, x_tp1 = q_sample_pairs(self.coeff, real_data[0].unsqueeze(0), t_emb)
      x_0_predict = self.g_net(x_tp1.detach(),t_emb, z)
      g_out = sample_posterior(self.pos_coeff, x_0_predict, x_tp1, t_emb)

      img_dim = g_out.view(g_out.size(0),-1).size(1)
   
      batch_size = self.optim_config.x_batch_size   
      z_dim = self.z_gen(1).size(1)
      print(img_dim)
      params = { 'proj.w': variance_scaling_init_(torch.Tensor(z_dim, 12288), scale=1.) }
      params1 = { 'proj.w': variance_scaling_init_(torch.Tensor(12288, 49152), scale=1.) }
      params2 = { 'proj.w': variance_scaling_init_(torch.Tensor(49152, img_dim), scale=1.) }
#       params = { 'proj.w': normal_(torch.Tensor(z_dim, 12288), std=0.01) }
#       params1 = { 'proj.w': normal_(torch.Tensor(12288, 49152), std=0.01) }
#       params2 = { 'proj.w': normal_(torch.Tensor(49152, img_dim), std=0.01) }
      # params1 = { 'proj.w': normal_(torch.Tensor(img_dim, img_dim), std=0.01) }
      # params1['proj.w'].requires_grad = True
#       params = { 'proj.w':  variance_scaling_init_(torch.Tensor(z_dim, img_dim),scale=1.)}
      params['proj.w'].requires_grad = True
      params1['proj.w'].requires_grad = True
      params2['proj.w'].requires_grad = True

      num_gened = 0
      fakes = torch.Tensor(cfg_N, img_dim)
      t_emb_s = torch.LongTensor(cfg_N)
      # print(t_emb_s.dtype)
      zs = torch.Tensor(cfg_N, z_dim)
      sz = [cfg_N] + list(g_out.size())[1:]
      x_t_s = torch.Tensor(torch.Size(sz), device=torch.device('cpu')) 
      x_tp1_s = torch.Tensor(torch.Size(sz), device=torch.device('cpu')) 
      with torch.no_grad():      
         while num_gened < cfg_N:
            num = min(batch_size, cfg_N - num_gened)
            z = self.z_gen(num)
            # z = z.to(self.device, non_blocking=True)
            # print(z.device)
            # print(params['proj.w'].device)
            # params['proj.w'] = params['proj.w'].to(self.device, non_blocking=True)
            # params1['proj.w'] = params1['proj.w'].to(self.device, non_blocking=True)
            fake = torch.mm(torch.mm(torch.mm(z, params['proj.w']),params1['proj.w']),params2['proj.w'])
            # fake =  torch.mm(fake1, params1['proj.w'])
            t_emb = torch.randint(0, self.time_step, (num,),device=self.device)
            # print(t_emb.dtype)

            t_emb_s[num_gened:num_gened+num] = t_emb
            sample,iterator = get_next(loader, iterator)    
            x_t, x_tp1 = q_sample_pairs(self.coeff, sample[0].to(self.device), t_emb)        
            x_t_s[num_gened:num_gened+num]=x_t.to(torch.device('cpu'))
            fakes[num_gened:num_gened+num] = fake
            zs[num_gened:num_gened+num] = z
            x_tp1_s[num_gened:num_gened+num]=x_tp1.to(torch.device('cpu'))
            num_gened += num
            
      to_pm1(fakes) # -> [-1,1]        
      # print(fakes)          
      dataset = TensorDataset(zs, fakes.view(sz),t_emb_s,x_t_s,x_tp1_s)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          pin_memory = torch.cuda.is_available())
      self._approximate(loader, g_loss)
         
   #-----------------------------------------------------------------
   def approximate(self, g_loss, cfg_N,sample_list): 
      if self.rank == 0:
        timeLog('DDG::approximate ... cfg_N=%d' % cfg_N)
      batch_size = self.optim_config.x_batch_size
#       num_gened = 0
#       sample_list = None
      target_fakes,zs,t_emb_s,x_t,x_tp_1= self.generate(cfg_N, do_return_z=True, batch_size=batch_size,real_data=sample_list,list_or_not=True)
      # print(target_fakes)
      dataset = TensorDataset(zs, target_fakes,t_emb_s,x_t,x_tp_1)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          pin_memory = torch.cuda.is_available())
      g_loss_v=self._approximate(loader, g_loss)
      return g_loss_v

   #-----------------------------------------------------------------
   def _approximate(self, loader, g_loss): 
      if self.verbose:
         timeLog('DDG::_approximate using %d data points ...' % len(loader.dataset))
      self.check_trainability()         
      optimizer = self.optim_config.create_optimizer(self.g_net.parameters(),use_ema=False,type='g')
      self.g_optimizer=optimizer
      mtr_loss = tnt.meter.AverageValueMeter()
      last_loss_mean = 99999999
      is_train = True
      # i = 0
      if self.rank == 0:
        timeLog('DDG::_approximate using %d data points ...' % len(loader.dataset))

      for epoch in range(self.optim_config.cfg_x_epo):
         # i =0
         for sample in loader:
            
            # timeLog('DDG::_approximate total epoch %d points ...' % i)
           
            z = cast(sample[0])
            z = z.to(self.device, non_blocking=True)
            target_fake = cast(sample[1])
            target_fake=target_fake.to(self.device, non_blocking=True)
            t_emb = sample[2].to(self.device,non_blocking=True)
            x_t= sample[3].to(self.device,non_blocking=True)
            x_tp1=sample[4].to(self.device,non_blocking=True)
            
           
            x_0_predict = self.g_net(x_tp1.detach(), t_emb, z)
            x_pos_sample = sample_posterior(self.pos_coeff, x_0_predict, x_tp1, t_emb)
            loss = g_loss(x_pos_sample, target_fake)
            mtr_loss.add(float(loss))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()                        
         loss_mean = mtr_loss.value()[0]
         if self.verbose:
            logging('%d ... %s ... ' % (epoch,str(loss_mean)))
         # logging('%d ... %s ... ' % (epoch,str(loss_mean)))   
         if loss_mean > last_loss_mean and self.rank == 0:
            self.optim_config.reduce_lr_(optimizer)
         raise_if_nan(loss_mean)

         last_loss_mean = loss_mean
         mtr_loss.reset()
      return loss_mean
#-----------------------------------------------------------------
def save_ddg(opt, ddg, stage):
   if not opt.save:
      return 
   
   stem = stem_name(opt.save, '.pth')
   pathname = stem + ('-stage%05d' % (stage+1)) + '.pth'
   ddg.save(opt, pathname)
def swap_ema(ddg):  
   ddg.optim_config.reduce_lr_(ddg.g_optimizer)

#-----------------------------------------------------------------
# data is [-1,1].  save_image expects [0,1]
def write_image(data, nm, nrow=None):
   # print(len(data))
   my_data = (data+1)/2  # [-1,1] -> [0,1]
   if nrow is not None:
      save_image(my_data, nm, nrow=nrow, pad_value=White,normalize=True)
   else:
      save_image(my_data, nm,normalize=True)

#-----------------------------------------------------------------
def generate(opt, ddg, stage='',l=1):
   if not opt.gen or opt.num_gen <= 0:
      return

   timeLog('Generating %d ... ' % opt.num_gen)
   stg = '-stg%05d' % (stage+1) if isinstance(stage,int) else str(stage)
   
   dir = os.path.dirname(opt.gen)
   if not os.path.exists(dir):
      os.makedirs(dir)   
   # num = 64   
   # x_t_1 = torch.randn_like(real_data)
   # sz = [opt.num_gen] + [3,256,256]
            # print(sz)
   # x_t_1 = torch.Tensor(torch.Size(sz), device=torch.device('cpu'))
   # x_t_1 = x_t_1.cuda()
   # fakes = sample_from_model(ddg.pos_coeff,ddg.g_net,ddg.g_params,ddg.time_step,x_t_1,None,ddg.z_gen(opt.num_gen).to(x_t_1.device))
   
    # num = 64   
   # x_t_1 = torch.randn_like(real_data)
   # if opt.image_size == 32:
   sz = [opt.num_gen] +[opt.num_channels]+[opt.image_size]+[opt.image_size]
   # print(sz)
   x_t_shape = torch.Tensor(torch.Size(sz), device=torch.device('cpu'))
   x_t_1 = torch.randn_like(x_t_shape)
   # x_t_1 = x_t_1.cuda()
   # print(time_t)
   fakes = sample_from_model_icfg(ddg.pos_coeff,opt.num_timesteps,x_t_1,ddg,opt)
   fake = fakes.to('cpu')  
   # x_t_1 = x_t_1.to('cpu')

   if opt.gen_nrow > 0:
      nm = opt.gen + '%s-%dc' % (stg,opt.num_gen) # 'c' for collage or collection
      write_image(fake, nm+'.jpg', nrow=opt.gen_nrow)   
   else:
      for i in range(opt.num_gen):
         nm = opt.gen +l+ ('%s-%d' % (stg,i))      
         write_image(fake[i], nm+'.jpg')
 
   timeLog('Done with generating %d ... ' % opt.num_gen)
       

#-------------------------------------------------------------
class OptimConfig:
   def __init__(self, opt):  
      self.verbose = opt.verbose
   
      #---  for discriminator and approximator    
      self.optim_type=opt.optim_type
      self.optim_eps=opt.optim_eps
      self.optim_a1=opt.optim_a1
      self.optim_a2=opt.optim_a2
      
      #---  for approximator 
      self.x_batch_size = opt.batch_size
      self.lr0 = opt.lr
      self.cfg_x_epo = opt.cfg_x_epo
      self.weight_decay = opt.weight_decay
      self.x_redmax = opt.approx_redmax # reduce lr if loss goes up, but do so only this many times. 
      self.x_decay = opt.approx_decay # to reduce lr, multiply this with lr. 

      self.redcount = 0
      self.lr = self.lr0
      self.use_ema = opt.use_ema
      self.ema_decay = opt.ema_decay
      self.lr_g = opt.lr_g
      self.lr_d = opt.lr_d
            
   def create_optimizer(self, params,use_ema=True,type='g'):
      self.redcount = 0
      if type == 'g':
         self.lr = self.lr_g
      if type =='d':
         self.lr = self.lr_d
      return create_optimizer(params, self.lr, self.optim_type, 
                              optim_eps=self.optim_eps, optim_a1=self.optim_a1, optim_a2=self.optim_a2, 
                              lam=self.weight_decay, verbose=self.verbose,use_ema=use_ema,ema_decay=self.ema_decay)
     
   def reduce_lr_(self,optimizer):
      # timeLog('reduce_lr_ x_redmax to '+str(self.x_redmax)+' redcount'+str(redcount)+'self.x_decay' +str(self.x_decay)+'in place ...')
      if self.x_decay <= 0:
         return
      if self.x_redmax > 0 and self.redcount >= self.x_redmax:
         return
      # timeLog('reduce_lr_ Setting before lr to '+str(lr)+' in place ...')
      self.lr *= self.x_decay
      # timeLog('reduce_lr_ Setting after lr to '+str(lr)+' in place ...')
      # timeLog('reduce_lr_ Setting lr to '+str(lr)+' in place ...')
      change_lr_(optimizer, self.lr, verbose=self.verbose)
      self.redcount += 1
   def swap_ema(self,optimizer):
      optimizer.swap_parameters_with_ema(store_params_in_ema=True)
      # return redcount,lr

#----------------------------------------------------------
def create_optimizer(params, lr, optim_type,  
                     optim_eps, optim_a1, optim_a2, 
                     lam, verbose,use_ema,ema_decay):
   optim_params =params
   if optim_type == RMSprop_str:
      alpha = optim_a1 if optim_a1 > 0 else 0.99  # pyTorch's default
      eps = optim_eps if optim_eps > 0 else 1e-8  # pyTorch's default
      msg = 'Creating RMSprop optimizer with lr='+str(lr)+', lam='+str(lam)+', alpha='+str(alpha)+', eps='+str(eps)
      optim = RMSprop(optim_params, lr, weight_decay=lam, alpha=alpha, eps=eps)
   elif optim_type == Adam_str:
      # NOTE: not tested.  
      eps = optim_eps if optim_eps > 0 else 1e-8  # pyTorch's default
      a1 = optim_a1 if optim_a1 > 0 else 0     # pyTorch's default
      a2 = optim_a2 if optim_a2 > 0 else 0.999    # pyTorch's default
      msg = 'Creating Adam optimizer with lr=%s, lam=%s, eps=%s, betas=(%s,%s)' % (str(lr),str(lam),str(eps),str(a1),str(a2))
      optim = Adam(optim_params, lr, betas=(a1,a2), eps=eps, weight_decay=lam)
   else:
      raise ValueError('Unknown optim_type: %s' % optim_type)
   if verbose:
      timeLog(msg)
   if use_ema:
        optim = EMA(optim, ema_decay=ema_decay)
   optim.zero_grad()
   return optim

#----------------------------------------------------------   
def change_lr_(optimizer, lr, verbose=False):
   # if verbose:
   timeLog('Setting lr to '+str(lr)+' in place ...')
   for param_group in optimizer.param_groups:
      param_group['lr'] = lr    

#-----------------------------------------------------------------
def check_opt_(opt):
   raise_if_absent(opt,['cfg_T','cfg_U','cfg_N','num_stages','batch_size','channels','lr','cfg_eta','cfg_x_epo','optim_type'], 'cfggan')
   add_if_absent_(opt, ['save','gen'], '')
   add_if_absent_(opt, ['save_interval','gen_interval','num_gen','approx_redmax','approx_decay','gen_nrow','diff_max'], -1)
   add_if_absent_(opt, ['optim_eps','optim_a1','optim_a2'], -1)
   add_if_absent_(opt, ['weight_decay'], 0.0)
   add_if_absent_(opt, ['verbose','do_exp'], False)

   raise_if_nonpositive_any(opt, ['cfg_T','cfg_U','cfg_N','num_stages','batch_size','channels','cfg_eta','lr','cfg_x_epo'])
    
#-----------------------------------------------------------------
def to_pm1(fake):
   fake.clamp_(-1,1)  