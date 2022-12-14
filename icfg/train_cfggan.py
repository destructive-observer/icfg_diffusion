import sys
import os
import argparse

from utils.utils0 import raise_if_absent, add_if_absent_, set_if_none, raise_if_nonpositive_any, show_args, ArgParser_HelpWithDefaults
from cfggan_train import proc as cfggan_train
from cfggan_train import DCGANx, Resnet4, FCn, Resnet3,DCGAN4,Balance,Resnet256,Resnet128,Resnet1024,Balance2
from cfggan import RMSprop_str

FC2 = 'fc2'
MNIST = 'MNIST'
SVHN = 'SVHN'
Bedroom64 = 'lsun_bedroom64'
ImageNet = 'ImageNet'
ImageNet64 = 'ImageNet64'
Church64 = 'lsun_church_outdoor64'
Brlr64 = 'lsun_brlr64'
Twbg64 = 'lsun_twbg64'
CIFAR10 = 'CIFAR10'
Tower64='lsun_tower64'
FashionMNIST = 'FashionMNIST'
EMNIST='EMNIST'
Bedroom128 = 'lsun_bedroom128'
Bedroom256 = 'lsun_bedroom256'
CelebaHQ = 'celebahq_1024'
Celeba128 = 'celebahq_128'
Celeba256 = 'celebahq_256'
#----------------------------------------------------------
def add_args_(parser):
   #---  proc
   parser.add_argument('--seed', type=int, default=1, help='Random seed.')   

   parser.add_argument('--dataset', type=str, choices=[MNIST, SVHN, Bedroom64, Church64,CIFAR10,Tower64,Brlr64,Twbg64,FashionMNIST,EMNIST,Bedroom256,Bedroom128,CelebaHQ,Celeba128,Celeba256,ImageNet,ImageNet64], required=True, help='Dataset.')
   parser.add_argument('--dataroot', type=str, default='.')
   parser.add_argument('--model', type=str, choices=[DCGANx,Resnet4,FC2,Resnet3,DCGAN4,Balance,Resnet256,Resnet128,Resnet1024,Balance2], help='Model.')   
   parser.add_argument('--norm_type', type=str, default='bn', choices=['bn','none'], help="'bn': batch normalization, 'none': no normalization")   
   parser.add_argument('--batch_size', type=int, default=64, help='Number of images for training.')   
   parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for retrieving images.')   

   #---  cfggan
   parser.add_argument('--cfg_T', type=int, help='T for ICFG.')
   parser.add_argument('--cfg_U', type=int, default=1, help='U (discriminator update frequency) for ICFG.')
   parser.add_argument('--cfg_N', type=int, default=640, help='N (number of generated examples used for approximator training).')  
   parser.add_argument('--num_stages', type=int, default=10000, help='Number of stages.')   
   parser.add_argument('--cfg_eta', type=float, help='Generator step-size eta.')
   parser.add_argument('--lr', type=float, help='Learning rate used for training discriminator and approximator.')
   parser.add_argument('--cfg_x_epo', type=int, default=10, help='Number of epochs for approximator training.')
      
   parser.add_argument('--gen', type=str, help='Pathname for saving generated images.')   
   parser.add_argument('--save', type=str, default='', help='Pathname for saving models.') 
   parser.add_argument('--save_interval', type=int, default=-1, help='Interval for saving models. -1: no saving.')    
   parser.add_argument('--gen_interval', type=int, default=50, help='Interval for generating images. -1: no generation.')  
   parser.add_argument('--num_gen', type=int, default=10, help='Number of images to be generated.')     
   parser.add_argument('--gen_nrow', type=int, default=5, help='Number of images in each row when making a collage of generated of images.')
   parser.add_argument('--diff_max', type=float, default=40, help='Stop training if |D(real)-D(gen)| exceeds this after passing the initial starting-up phase.')
   parser.add_argument('--lamda',type=float ,default='0', help='0 stands for no constraint or others stands for lamda value')
   parser.add_argument('--scale',type=float ,default='0.1', help='scale for the eposion, value is [0,1]')
   parser.add_argument('--gptype', type=int, default='0', help='0-0 centered 1-1 centered 2-newtype.') 
   parser.add_argument('--alpha',type=float ,default='0.9', help='use w distance to regulation regression')
   parser.add_argument('--verbose', action='store_true', help='If true, display more info.')   
   parser.add_argument('--saved', type=str, default='', help='Pathname for the saved model.') 
#----------------------------------------------x------------
def check_args_(opt):
   if opt.batch_size is None:
      opt.batch_size = 64 # Batch size. 
   opt.z_dim = 100 # Dimensionality of input random vectors.
   opt.z_std = 1.0 # Standard deviation for generating input random vectors.
   opt.approx_redmax = 5
   opt.approx_decay = 0.1
   opt.cfg_N = opt.batch_size * opt.cfg_x_epo
   
   def is_32x32():
      return opt.dataset in [MNIST, SVHN]   
   def is_32x32_monocolor():
      return opt.dataset == MNIST
      
   #***  Setting meta-parameters to those used in the CFG-GAN paper. 
   #---  network architecture, learning rate, and T
   if opt.model is None:
      opt.model = Resnet4 if opt.dataset.endswith('64') else DCGANx
      
   if opt.model == DCGANx:
      opt.d_model = opt.g_model = DCGANx
      opt.d_depth = opt.g_depth = 3 if is_32x32() else 4
      opt.d_dim = opt.g_dim = 32 if is_32x32_monocolor() else 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == Resnet4:
      opt.d_model = opt.g_model = Resnet4
      opt.d_depth = opt.g_depth = 4
      opt.d_dim = opt.g_dim = 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)   
   elif opt.model == FC2:
      opt.d_model = DCGANx
      opt.d_dim = 32 if is_32x32_monocolor() else 64
      opt.d_depth = 3 if is_32x32() else 4
      opt.g_model = FCn; opt.g_depth = 2; opt.g_dim = 512
      set_if_none(opt, 'lr', 0.0001)
      set_if_none(opt, 'cfg_T', 25)
   elif opt.model == Resnet3:
      opt.d_model = opt.g_model = Resnet3
      opt.d_depth = opt.g_depth = 3
      opt.d_dim = opt.g_dim = 32
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == DCGAN4:
      opt.d_model = opt.g_model = DCGANx
      opt.d_depth = opt.g_depth = 3 if is_32x32() else 4
      opt.d_dim = opt.g_dim = 32 if is_32x32_monocolor() else 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == Balance2:
      opt.d_model = Resnet4
      opt.g_model = DCGANx
      opt.d_depth = 2
      opt.g_depth = 3
      opt.d_dim = 32
      opt.g_dim = 32
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)   
   elif opt.model == Balance:
      opt.d_model = Resnet4
      opt.g_model = DCGANx
      opt.d_depth = 3
      opt.g_depth = 4
      opt.d_dim = 64
      opt.g_dim = 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == Resnet256:
      opt.d_model = opt.g_model = Resnet4
      opt.d_depth = 6
      opt.g_depth = 5
      opt.d_dim = opt.g_dim = 64
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)    
   elif opt.model == Resnet128:
      opt.d_model = opt.g_model = Resnet4
      opt.d_depth = 5
      opt.g_depth = 4
      opt.d_dim = opt.g_dim = 128
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)
   elif opt.model == Resnet1024:
      opt.d_model = Resnet4
      opt.g_model = Resnet1024
      opt.d_depth = 8
      opt.g_depth = 7
      opt.d_dim = opt.g_dim = 32
      set_if_none(opt, 'lr', 0.00025)
      set_if_none(opt, 'cfg_T', 15)         
   else:
      raise ValueError('Unknown model: %s' % opt.model)

   #---  eta (generator step-size)
   if opt.cfg_eta is None:
      if opt.dataset == MNIST:
         dflt = { DCGANx+'bn': 0.5,  DCGANx+'none': 2.5, FC2+'bn': 0.1 }
      elif opt.dataset == SVHN:
         dflt = { DCGANx+'bn': 0.25, DCGANx+'none': 0.5, FC2+'bn': 0.5 }
      elif opt.dataset == CIFAR10:
         dflt = { DCGANx+'bn': 0.25, DCGANx+'none': 0.5, FC2+'bn': 0.5 }
      else:
         dflt = { Resnet4+'bn': 1, Resnet4+'none': 2.5, FC2+'bn': 0.5 }
      
      opt.cfg_eta = dflt.get(opt.model+opt.norm_type)
      if opt.cfg_eta is None:
         raise ValueError("'cfg_eta' is missing.")

   #---  optimization 
   opt.optim_type=RMSprop_str; opt.optim_eps=1e-18; opt.optim_a1=0.95; opt.optim_a2=-1
   # RMSprop used in the paper adds epsilon *before* sqrt, but pyTorch does 
   # this *after* sqrt, and so this setting is close to but not exactly the same as the paper.  
   # Adam or RMSprop with pyTorch default values can be used too, but 
   # learning rate may have to be re-tuned.
      
   #***  Setting default values for generating examples
   if opt.gen is None and opt.num_gen > 0 and opt.gen_interval > 0:
      dir = 'gen'
      if not os.path.exists(dir):
         os.mkdir(dir)
      opt.gen = dir + os.path.sep + opt.dataset + '-' + opt.model
 
   if opt.save:
      if opt.save_interval is None or opt.save_interval <= 0:
         opt.save_interval = 100
   opt.logstr = opt.gen+'-'+opt.dataset+'-'+'opt.cfg_eta'+str(opt.cfg_eta)+'-'+'opt.lr'+str(opt.lr)
   #***  Display arguments 
   show_args(opt, ['dataset','dataroot','num_workers','logstr'])
   raise_if_nonpositive_any(opt, ['d_dim','g_dim','z_dim','z_std'])
   show_args(opt, ['d_model','d_dim','d_depth','g_model','g_dim','g_depth','norm_type','z_dim','z_std'], 'Net definitions ----')
   raise_if_nonpositive_any(opt, ['cfg_T','cfg_U','cfg_N','batch_size','num_stages','cfg_eta','lr','cfg_x_epo'])
   show_args(opt, ['cfg_T','cfg_U','cfg_N','num_stages','cfg_eta','cfg_x_epo','diff_max','lamda','alpha','gptype'], 'CFG --- ')
   show_args(opt, ['optim_type','optim_eps','optim_a1','optim_a2','lr','batch_size'], 'Optimization ---')
   show_args(opt, ['seed','gen','save','save_interval','gen_interval','num_gen','gen_nrow','verbose'], 'Others ---')
   ## log file to store the param##
      ## log file to store the param##
   logfile = os.path.dirname(opt.gen)
   print('logstr {}'.format(opt.logstr))
      #dir = os.path.dirname(logfile)
   if not os.path.exists(logfile):
      os.makedirs(logfile)
   logfile = logfile+os.path.sep+'log.txt'
   with open(logfile,'w') as f:
      od = vars(opt)
      for k,v in od.items():
         f.write(str(k)+':'+str(v)+'\n')
      #logfile = dir+os.path.sep+'log.txt'
#----------------------------------------------------------
def main(args):
   parser = ArgParser_HelpWithDefaults(description='cfggan_train', formatter_class=argparse.MetavarTypeHelpFormatter)
   add_args_(parser)
   opt = parser.parse_args(args) 
   check_args_(opt)
   cfggan_train(opt)
  
if __name__ == '__main__':
   main(sys.argv[1:])   
