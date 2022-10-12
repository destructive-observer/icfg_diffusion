import argparse
import torch
from torch.backends import cudnn
from torch.nn.init import normal_
import numpy as np

from cfggan_train import DCGANx, Resnet4, FCn,DCGAN4,Resnet3
import netdef
from cfggan import DDG, generate
from utils.utils0 import raise_if_absent, add_if_absent_, raise_if_nonpositive_any, show_args, ArgParser_HelpWithDefaults, timeLog

cudnn.benchmark = True

#----------------------------------------------------------
def main():
   parser = ArgParser_HelpWithDefaults(description='cfggan_gen', formatter_class=argparse.MetavarTypeHelpFormatter)
   parser.add_argument('--seed', type=int, default=1, help='Random seed.')   
   parser.add_argument('--gen', type=str, required=True, help='Pathname for writing generated images.')   
   parser.add_argument('--saved', type=str, required=True, help='Pathname for the saved model.') 
   parser.add_argument('--num_gen', type=int, default=40, help='Number of images to be generated.') 
   parser.add_argument('--gen_nrow', type=int, default=8, help='Number of images in each row when making a collage. -1: No collage.')

   gen_opt = parser.parse_args() 
   show_args(gen_opt, ['seed','gen','saved','num_gen'])

   torch.manual_seed(gen_opt.seed)
   np.random.seed(gen_opt.seed)
   if torch.cuda.is_available():
      torch.cuda.manual_seed_all(gen_opt.seed)

   torch.backends.cudnn.benchmark = True
   timeLog('Reading from %s ... ' % gen_opt.saved)
   from_file = torch.load(gen_opt.saved, map_location=None if torch.cuda.is_available() else 'cpu')

   opt = from_file['opt']
   print(opt)
   print(hasattr(opt,'gptype'))
   if hasattr(opt,'gptype') == False:
      opt.gptype = 2
   #---  these must be in sync with cfggan_train  --------------
   def d_config(requires_grad):  # D
      if opt.d_model == DCGANx:
         return netdef.dcganx_D(opt.d_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.d_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.d_model == DCGAN4:
         return netdef.dcganx_D(opt.d_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.d_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.d_model == Resnet4:
         return netdef.resnet4_D(opt.d_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad,  depth=opt.d_depth, 
                                 do_bias=not opt.do_no_bias)
      elif opt.d_model == Resnet3:
         # if opt.d_depth != 3:
            # logging('WARNING: d_depth is ignored as d_model is Resnet3.')
         return netdef.resnet4_D(opt.d_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad, depth=opt.d_depth,  
                                 do_bias=not opt.do_no_bias)      
      else:
         raise ValueError('Unknown d_model: %s' % opt.d_model)
   def g_config(requires_grad):  # G
      if opt.g_model == DCGANx:
         return netdef.dcganx_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.g_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.g_model == Resnet4:   
         return netdef.resnet4_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad, depth=opt.g_depth,  do_bias=not opt.do_no_bias)
      elif opt.g_model == DCGAN4:
         return netdef.dcganx_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                opt.norm_type, requires_grad, depth=opt.g_depth, 
                                do_bias=not opt.do_no_bias)
      elif opt.g_model == Resnet3:
         # if opt.g_depth != 3:
         #    logging('WARNING: d_depth is ignored as d_model is Resnet3.')      
         return netdef.resnet4_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, 
                                 opt.norm_type, requires_grad, depth=opt.g_depth,  do_bias=not opt.do_no_bias)                                     
      elif opt.g_model == FCn:
         return netdef.fcn_G(opt.z_dim, opt.g_dim, opt.image_size, opt.channels, requires_grad, depth=opt.g_depth)
      else:
         raise ValueError('Unknown g_model: %s' % opt.g_model)
   def z_gen(num):
      return normal_(torch.Tensor(num, opt.z_dim), std=opt.z_std)
   #-------------------------------------------------------------

   ddg = DDG(opt, d_config, g_config, z_gen, optim_config=None, from_file=from_file)
   for i in range(20):
      # gen_opt["num_gen"]=40*i
      generate(gen_opt, ddg,l=str(i))

#----------------------------------------------------------
def check_opt_(opt):
   raise_if_absent(opt, ['saved','num_gen','gen','gen_nrow'], 'cfggan_gen')
   add_if_absent_(opt, ['seed'], 1)

   raise_if_nonpositive_any(opt, ['num_gen'])  

#----------------------------------------------------------
if __name__ == '__main__':
   main() 