import os,time
import torch
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader
import torchnet as tnt
from torch.optim import RMSprop, Adam, SGD
from torchvision.utils import save_image

from utils.utils import cast
from utils.utils0 import timeLog, copy_params, clone_params, print_params, print_num_params, stem_name
from utils.utils0 import raise_if_absent, add_if_absent_, logging, raise_if_nonpositive_any, raise_if_nan
from torch.autograd import Variable
import numpy as np
# from logger import Logger
# import visdom
import math
White=255
RMSprop_str='RMSprop'
Adam_str='Adam'
# vizG = visdom.Visdom(env='G2')  # 初始化visdom类
# vizD = visdom.Visdom(env='D2')
# vizG_n = visdom.Visdom(env='Gn')
# vizG_f = visdom.Visdom(env='fn')
#-----------------------------------------------------------------
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
def wgan_gp(self,fake,real,LAMBDA,netD,d_param,centered):
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
   disc_interpolates = netD(interpolates,d_param,True)
   gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),#.cuda(), #if use_cuda else torch.ones(
                                    #disc_interpolates.size()),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]#LAMBDA = 1
            # print('gradients.size {}'.format(gradients.shape)
   gradients = gradients.reshape(gradients.size(0),-1)
   gradient_penalty = ((gradients.norm(2, dim=1) - centered) ** 2).mean() * LAMBDA
        
   return gradient_penalty
def new_gp(self,fake,real,LAMBDA,netD,d_param,centered,scale=0.1):
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
   disc_interpolates = netD(interpolates,d_param,True)
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

   check_opt_(opt)
 
   write_real(opt, loader)
   # write_real_num(opt, loader)


   optim_config = OptimConfig(opt)
   begin = 0
   ddg = DDG(opt, d_config, g_config, z_gen, optim_config,fromfile)
   timeLog('saved ' + str(saved) + '.')
   # if fromfile != None:
   #    ddg.initialize_G(g_loss, opt.cfg_N)
   if saved:
      begin = saved[-8:-4]
      timeLog('stages ' + str(begin) + '.')
      begin = int(begin)
   else:
      ddg.initialize_G(g_loss, opt.cfg_N)

   #---  xICFG
   iterator = None
   for stage in range(begin,opt.num_stages):   
      timeLog('xICFG stage %d -----------------' % (stage+1))
      iterator,diff,d_loss_v,d_loss_gp,d_fake,d_real = ddg.icfg(loader, iterator, d_loss, opt.cfg_U)
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
         save_ddg(opt, ddg, stage)
      if is_time_to_generate(opt, stage):
         generate(opt, ddg, stage)
         
      g_loss_v=ddg.approximate(g_loss, opt.cfg_N)         
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
      self.d_params_list = [ d_config(requires_grad=False)[1] for i in range(opt.cfg_T) ]
      # print('d_params_list {}'.format(self.d_params_list))
      self.d_net,self.d_params = d_config(requires_grad=True)
      self.g_net,self.g_params = g_config(requires_grad=True)
      self.z_gen = z_gen
      self.cfg_eta = opt.cfg_eta
      self.alpha = opt.alpha
      self.optim_config = optim_config
      self.d_optimizer = None
      self.current_time = time.strftime('%Y-%m-%d %H%M%S')
      self.logstr = opt.logstr
      self.lamda = opt.lamda
      self.gptype = opt.gptype
      self.scale = opt.scale
      self.epoch = 0
      # self.logger = Logger('./logs/' + str(opt.gen)+'-' +str(opt.cfg_eta)+ "/")
      if optim_config is not None:
         self.d_optimizer = optim_config.create_optimizer(self.d_params)

      if from_file is not None:
         self.load(from_file)

      logging('----  D  ----')
      if self.verbose:         
         print_params(self.d_params)       
      print_num_params(self.d_params) 
      
      logging('----  G  ----')
      if self.verbose:
         print_params(self.g_params)
      print_num_params(self.g_params)          

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
                      d_params=self.d_params,
                      g_params=self.g_params,
                      cfg_eta=self.cfg_eta,
                      opt=opt), 
                 path)
                 
   def load(self,  d):
      assert len(self.d_params_list) == len(d['d_params_list'])
      for i in range(len(self.d_params_list)):
         copy_params(src=d['d_params_list'][i], dst=self.d_params_list[i])
      copy_params(src=d['d_params'], dst=self.d_params)
      copy_params(src=d['g_params'], dst=self.g_params)      
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
      copy_params(src=self.d_params, dst=self.d_params_list[t])
      
   #----------------------------------------------------------      
   def generate(self, num_gen, t=-1, do_return_z=False, batch_size=-1):
      assert num_gen > 0
      if t < 0:
         t = self.num_D()
      if batch_size <= 0:
         batch_size = num_gen
         
      num_gened = 0
      fakes = None
      zs = None
      is_train = False
      while num_gened < num_gen:
         num = min(batch_size, num_gen - num_gened)
         with torch.no_grad():
            z = self.z_gen(num)
            fake = self.g_net(cast(z), self.g_params, is_train)
         # if self.verbose:   
         #    if self.epoch % 50 == 0:
         #       vizG_n.images(fake,opts=dict(title='-1 - 1 fake images vizG_n+stage{}+numD xxxx'.format(self.epoch), caption='vizG_n D.'))   
         #       fake_g = (fake+1)/2
         #       vizG_n.images(fake_g,opts=dict(title='0 - 1 fake images vizG_n+stage{}+numD xxxx'.format(self.epoch), caption='vizG_n D.'))    

         for t0 in range(t):
            fake = fake.detach()
            if fake.grad is not None:
               fake.grad.zero_()
            fake.requires_grad = True
      
            d_out = self.d_net(fake, self.get_d_params(t0), True)
            d_out_1 = d_out.view(-1,d_out.size(0))
            if self.verbose:
               timeLog('DDG::generate ... with d_out={}'.format(d_out_1))
            # d_out = self.d_net(fake, self.d_params, True)
            d_out.backward(torch.ones_like(d_out))   
            fake.data += self.cfg_eta * fake.grad.data
            # vizG_n.images(fake.grad.data,opts=dict(title='-1 - 1 fake  grad images vizG_n+stage{}+numD {}'.format(self.epoch,t0), caption='vizG_n D.'))   
            # fake_g = (fake.grad.data+1)/2
            # vizG_n.images(fake_g,opts=dict(title='0 - 1 fake grad images vizG_n+stage{}+numD {}'.format(self.epoch,t0), caption='vizG_n D.'))    
            # print('fake data shape {}'.format(fake.data.shape))
            if self.verbose:
               timeLog('DDG::generate ... with fake.data=%f and fake.grad=%f' % (torch.sum(fake.data),torch.sum(fake.grad.data)))
            # fake.data += self.cfg_eta * (fake.grad.data*torch.mean(self.real_sample)-torch.mean(d_out))/torch.mean(self.real_sample)/torch.mean(self.real_sample)
         # if self.verbose:
         #    if self.epoch % 50 == 0:
         #       vizG_n.images(fake,opts=dict(title='-1 - 1 fake images vizG_n+stage{}+numD full'.format(self.epoch), caption='vizG_n D.'))   
         #       fake_g = (fake+1)/2
         #       vizG_n.images(fake_g,opts=dict(title='0 - 1 fake images vizG_n+stage{}+numD full'.format(self.epoch), caption='vizG_n D.')) 
         if fakes is None:
            sz = [num_gen] + list(fake.size())[1:]
            fakes = torch.Tensor(torch.Size(sz), device=torch.device('cpu'))

         fakes[num_gened:num_gened+num] = fake.to(torch.device('cpu'))

         if do_return_z:
            if zs is None:  
               sz = [num_gen] + list(z.size())[1:]            
               zs = torch.Tensor(torch.Size(sz), device=z.device)
            zs[num_gened:num_gened+num] = z
 
         num_gened += num

      fakes.detach_()
      if do_return_z:
         return fakes, zs
      else:
         return fakes

   #-----------------------------------------------------------------
   def icfg(self, loader, iter, d_loss, cfg_U):   
      timeLog('DDG::icfg ... ICFG with cfg_U=%d' % cfg_U)
      timeLog('DDG::icfg ... ICFG with settings=%s' % self.logstr)
      self.check_trainability()
      t_inc = 1 if self.verbose else 5
      is_train = True
      for t in range(self.num_D()):
         # print('self.num_D {}'.format(self.num_D()))
         sum_real = sum_fake = count = 0
         for upd in range(cfg_U):
            sample,iter = get_next(loader, iter)

            num = sample[0].size(0)
            # print('num is {}'.format(sample[0]))
            self.real_sample = sample[0]
            fake = self.generate(num, t=t)
            # if self.verbose:
            #    if self.epoch % 50 ==0:           
            #       vizD.images(fake,opts=dict(title='-1 - 1 fake images D+stage{}+numD {}'.format(self.epoch,t), caption='fake D.'))   
            #       fake_g = (fake+1)/2
            #       vizD.images(fake_g,opts=dict(title='0 - 1 fake images D+stage{}+numD{}'.format(self.epoch,t), caption='fake D.'))           
            # # vizD.images(sample[0],opts=dict(title='real images D', caption='real.D'))
            d_out_real = self.d_net(cast(sample[0]), self.d_params, is_train)
            d_out_fake = self.d_net(cast(fake), self.d_params, is_train)            
            # d_out_real = self.d_net(cast(sample[0]), self.get_d_params(t), is_train)
            # d_out_fake = self.d_net(cast(fake), self.get_d_params(t), is_train)
            # loss = d_loss(d_out_real, d_out_fake)
            loss1,loss2 = d_loss(d_out_real, d_out_fake,self.alpha)
            loss = loss1 + loss2
            loss.backward()
            loss_gp=0
            # print(self.lamda)
            if self.lamda != 0:
               # timeLog('DDG::icfg ... ICFG with lamda=%s' % str(self.lamda))
               # loss_gp = wgan_gp(self,fake,sample[0],self.lamda,self.d_net,self.d_params)
               # print(self.lamda)
               if self.gptype ==0:
                  # print('0 ----{}'.format(self.gptype))
                  loss_gp = wgan_gp(self,fake,sample[0],self.lamda,self.d_net,self.d_params,0)
                  loss_gp.backward()
               elif self.gptype ==1:
                  # print('1 ----{}'.format(self.gptype))
                  loss_gp = wgan_gp(self,fake,sample[0],self.lamda,self.d_net,self.d_params,1)
                  loss_gp.backward()
               elif self.gptype ==2:
                  # print('2 ----{}'.format(self.gptype))
                  loss_gp = new_gp(self,fake,sample[0],self.lamda,self.d_net,self.d_params,0,scale=self.scale)
                  loss_gp.backward()
               else:
                  raise ValueError('Unknown gptype: %s ...' % self.gptype)

            self.d_optimizer.step()
            self.d_optimizer.zero_grad()         
            
            with torch.no_grad():
               sum_real += float(d_out_real.sum()); sum_fake += float(d_out_fake.sum()); count += num            
            
         self.store_d_params(t)
         
         if t_inc > 0 and ((t+1) % t_inc == 0 or t == self.num_D()-1):
            logging('  t=%d: real,%s, fake,%s ' % (t+1, sum_real/count, sum_fake/count))
            logging('  t=%d: loss_logistic,%s, loss_wgan,%s ' % (t+1, loss1, loss2))

      raise_if_nan(sum_real)
      raise_if_nan(sum_fake)

      return iter,(sum_real-sum_fake)/count,loss,loss_gp,sum_fake/count,sum_real/count

   #-----------------------------------------------------------------
   def initialize_G(self, g_loss, cfg_N): 
      timeLog('DDG::initialize_G ... Initializing tilde(G) ... ')
      z = self.z_gen(1)
      g_out = self.g_net(cast(z), self.g_params, False)
      img_dim = g_out.view(g_out.size(0),-1).size(1)
   
      batch_size = self.optim_config.x_batch_size   
      z_dim = self.z_gen(1).size(1)
      params = { 'proj.w': normal_(torch.Tensor(z_dim, img_dim), std=0.01) }
      params['proj.w'].requires_grad = True
         
      num_gened = 0
      fakes = torch.Tensor(cfg_N, img_dim)
      zs = torch.Tensor(cfg_N, z_dim)
      with torch.no_grad():      
         while num_gened < cfg_N:
            num = min(batch_size, cfg_N - num_gened)
            z = self.z_gen(num)
            fake = torch.mm(z, params['proj.w'])
            fakes[num_gened:num_gened+num] = fake
            zs[num_gened:num_gened+num] = z
            num_gened += num
            
      to_pm1(fakes) # -> [-1,1]            
            
      sz = [cfg_N] + list(g_out.size())[1:]
      dataset = TensorDataset(zs, fakes.view(sz))
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          pin_memory = torch.cuda.is_available())
      self._approximate(loader, g_loss)
         
   #-----------------------------------------------------------------
   def approximate(self, g_loss, cfg_N): 
      timeLog('DDG::approximate ... cfg_N=%d' % cfg_N)
      batch_size = self.optim_config.x_batch_size
      target_fakes,zs = self.generate(cfg_N, do_return_z=True, batch_size=batch_size)
      dataset = TensorDataset(zs, target_fakes)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          pin_memory = torch.cuda.is_available())
                          
      g_loss_v=self._approximate(loader, g_loss)
      return g_loss_v

   #-----------------------------------------------------------------
   def _approximate(self, loader, g_loss): 
      if self.verbose:
         timeLog('DDG::_approximate using %d data points ...' % len(loader.dataset))
      self.check_trainability()         
      with torch.no_grad():
         g_params = clone_params(self.g_params, do_copy_requires_grad=True)

      optimizer = self.optim_config.create_optimizer(g_params)
      mtr_loss = tnt.meter.AverageValueMeter()
      last_loss_mean = 99999999
      is_train = True
      # i = 0
      timeLog('DDG::_approximate using %d data points ...' % len(loader.dataset))
      for epoch in range(self.optim_config.cfg_x_epo):
         # i =0
         for sample in loader:
            
            # timeLog('DDG::_approximate total epoch %d points ...' % i)
           
            z = cast(sample[0])
            target_fake = cast(sample[1])
            # if i == 0:
            # vizG_f.images(target_fake,opts=dict(title='-1 - 1 epoch target_fake images{}'.format(epoch), caption='G fake.'))           
            # fake_g = (target_fake+1)/2
            # vizG_f.images(fake_g,opts=dict(title='0-1 G epoch target_fake images{}'.format(epoch), caption='G fake.'))           
            # i=i+1
            fake = self.g_net(z, g_params, is_train)

            loss = g_loss(fake, target_fake)
            mtr_loss.add(float(loss))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()                        
         loss_mean = mtr_loss.value()[0]
         if self.verbose:
            logging('%d ... %s ... ' % (epoch,str(loss_mean)))
         # logging('%d ... %s ... ' % (epoch,str(loss_mean)))   
         if loss_mean > last_loss_mean:
            # logging('loss_mean%d ... %s ... ' % (epoch,str(loss_mean)))
            # logging('last_loss_mean%d ... %s ... ' % (epoch,str(last_loss_mean)))
            # logging('last_loss_mean%d ... %s ... %s...' % (epoch,str(self.optim_config.redcount),str(self.optim_config.lr)))
            self.optim_config.reduce_lr_(optimizer)
            # vizG.images(fake,opts=dict(title='-1 - 1 G fake images+{}'.format(epoch), caption='G fake.'))           
            # fake_g = (fake+1)/2
            # vizG.images(fake_g,opts=dict(title='0-1 G fake images+{}'.format(epoch), caption='G fake.'))

            # vizG.images(target_fake,opts=dict(title='-1 - 1 G target_fake images+{}'.format(epoch), caption='G target_fake.'))
            # target_fake_g = (target_fake+1)/2            
            # vizG.images(target_fake_g,opts=dict(title='0-1 G target_fake images+{}'.format(epoch), caption='G target_fake.'))
            # logging('last_loss_mean%d ... %s ... %s...' % (epoch,str(self.optim_config.redcount),str(self.optim_config.lr)))
         raise_if_nan(loss_mean)

         last_loss_mean = loss_mean
         mtr_loss.reset()
      # if self.verbose:
         # vizG.images(fake,opts=dict(title='-1 - 1 G fake images', caption='G fake.'))           
         # fake_g = (fake+1)/2
         # vizG.images(fake_g,opts=dict(title='0-1 G fake images', caption='G fake.'))
         # vizG.images(target_fake,opts=dict(title='-1 - 1 G target_fake images', caption='G target_fake.'))
         # target_fake_g = (target_fake+1)/2            
         # vizG.images(target_fake_g,opts=dict(title='0-1 G target_fake images', caption='G target_fake.'))               
      copy_params(src=g_params, dst=self.g_params)
      return loss_mean

#-----------------------------------------------------------------
def save_ddg(opt, ddg, stage):
   if not opt.save:
      return 
   
   stem = stem_name(opt.save, '.pth')
   pathname = stem + ('-stage%05d' % (stage+1)) + '.pth'
   ddg.save(opt, pathname)

#-----------------------------------------------------------------
# data is [-1,1].  save_image expects [0,1]
def write_image(data, nm, nrow=None):
   # print(len(data))
   my_data = (data+1)/2  # [-1,1] -> [0,1]
   if nrow is not None:
      save_image(my_data, nm, nrow=nrow, pad_value=White)
   else:
      save_image(my_data, nm)

#-----------------------------------------------------------------
def generate(opt, ddg, stage='',l=1):
   if not opt.gen or opt.num_gen <= 0:
      return

   timeLog('Generating %d ... ' % opt.num_gen)
   stg = '-stg%05d' % (stage+1) if isinstance(stage,int) else str(stage)
   
   dir = os.path.dirname(opt.gen)
   if not os.path.exists(dir):
      os.makedirs(dir)   
      
   fake = ddg.generate(opt.num_gen)

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
            
   def create_optimizer(self, params):
      self.redcount = 0
      self.lr = self.lr0   
      return create_optimizer(params, self.lr0, self.optim_type, 
                              optim_eps=self.optim_eps, optim_a1=self.optim_a1, optim_a2=self.optim_a2, 
                              lam=self.weight_decay, verbose=self.verbose)
     
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
      # return redcount,lr

#----------------------------------------------------------
def create_optimizer(params, lr, optim_type,  
                     optim_eps, optim_a1, optim_a2, 
                     lam, verbose):
   optim_params = [ {'params': [ v for k,v in sorted(params.items()) if v.requires_grad ]} ]
   if optim_type == RMSprop_str:
      alpha = optim_a1 if optim_a1 > 0 else 0.99  # pyTorch's default
      eps = optim_eps if optim_eps > 0 else 1e-8  # pyTorch's default
      msg = 'Creating RMSprop optimizer with lr='+str(lr)+', lam='+str(lam)+', alpha='+str(alpha)+', eps='+str(eps)
      optim = RMSprop(optim_params, lr, weight_decay=lam, alpha=alpha, eps=eps)
   elif optim_type == Adam_str:
      # NOTE: not tested.  
      eps = optim_eps if optim_eps > 0 else 1e-8  # pyTorch's default
      a1 = optim_a1 if optim_a1 > 0 else 0.9      # pyTorch's default
      a2 = optim_a2 if optim_a2 > 0 else 0.999    # pyTorch's default
      msg = 'Creating Adam optimizer with lr=%s, lam=%s, eps=%s, betas=(%s,%s)' % (str(lr),str(lam),str(eps),str(a1),str(a2))
      optim = Adam(optim_params, lr, betas=(a1,a2), eps=eps, weight_decay=lam)
   else:
      raise ValueError('Unknown optim_type: %s' % optim_type)
      
   if verbose:
      timeLog(msg)
      
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