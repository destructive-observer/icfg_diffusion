import numpy as np
import torchvision.transforms as T
import torchvision.datasets as datasets
# import torchvision.IterableDataset as IterableDataset
from torch.utils.data import ConcatDataset, Subset
from icfg.utils.utils0 import logging, timeLog

import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
#----------------------------------------------------------
def get_ds_attr(dataset):
   imgsz = 32; channels = 3
   if dataset == 'CIFAR10' or dataset == 'SVHN':
      nclass = 10
   elif dataset == 'CIFAR100':
      nclass = 100
   elif dataset == 'ImageNet': 
      nclass = 1000; imgsz = 224
   elif dataset == 'ImageNet64': 
      nclass = 1000; imgsz = 64
   elif dataset == 'MNIST':
      nclass = 10; channels = 1
   elif dataset  == 'FashionMNIST':
      nclass = 10; channels = 1
   elif dataset  == 'EMNIST':
      nclass = 10; channels = 1   
   elif dataset.endswith('64'):
      nclass = 1; imgsz = 64
      if 'brlr' in dataset or 'twbg' in dataset:
         nclass = 2
   elif dataset.endswith('256'):
      nclass = 1; imgsz = 256
      if 'brlr' in dataset or 'twbg' in dataset:
         nclass = 2
   elif dataset.endswith('128'):
      nclass = 1; imgsz = 128
      if 'brlr' in dataset or 'twbg' in dataset:
         nclass = 2
   elif dataset.endswith('1024'):
      nclass = 1; imgsz = 1024         
   else:
      raise ValueError('Unknown dataset: %s ...' % dataset)

   return { "nclass": nclass, "image_size": imgsz, "channels": channels }

#----------------------------------------------------------
def gen_lsun_balanced(dataroot, nms, tr, indexes):
   sub_dss = []
   for i,nm in enumerate(nms):
      sub_dss += [Subset(datasets.LSUN(dataroot, classes=[nm], transform=tr), indexes)]
   return ConcatUniClassDataset(sub_dss)

#----------------------------------------------------------
# Concatenate uni-class datasets into one dataset. 
class ConcatUniClassDataset:
   def __init__(self, dss):
      self.dss = dss
      self.top = [0]
      num = 0
      for ds in self.dss:
         num += len(ds)
         self.top += [num]
         
   def __len__(self):
      # print(self.top[len(self.top)-1])
      return self.top[len(self.top)-1]
         
   def __getitem__(self, index):
      cls = -1
      for i,top in enumerate(self.top):
         # print('i {} and top{} and index{}'.format(i,top,index))
         if index < top:
            cls = i-1
            break
      if cls < 0:
         raise IndexError
      # print((self.dss[cls])[index-top][0].shape)
      return ((self.dss[cls])[index-top][0], cls)
         
#----------------------------------------------------------
def get_ds(dataset, dataroot, is_train, do_download, do_augment):
   tr = get_tr(dataset, is_train, do_augment)
   if dataset == 'SVHN':
      if is_train:
         train_ds = datasets.SVHN(dataroot, split='train', transform=tr, download=do_download)
         extra_ds = datasets.SVHN(dataroot, split='extra', transform=tr, download=do_download)
         return ConcatDataset([train_ds, extra_ds])
      else:
         return datasets.SVHN(dataroot, split='test', transform=tr, download=do_download) 
   elif dataset == 'MNIST':
      return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download)    
   elif dataset =='CIFAR10':
      return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download)
   elif dataset =='FashionMNIST':
      return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download)
   elif dataset.startswith('ImageNet'):
      return ImageFolder(dataroot,transform=tr)   
   elif dataset == 'EMNIST':
      return getattr(datasets, dataset)(dataroot, train=is_train, transform=tr, download=do_download,split='byclass')
   elif dataset.startswith('celebahq_'):
      return ImageFolder(dataroot,transform=tr)               
   elif dataset.startswith('lsun_'):
      if  dataset.endswith('64'):
         nm = dataset[len('lsun_'):len(dataset)-len('64')] + ('_train' if is_train else '_val')
      elif dataset.endswith('256'):
         nm = dataset[len('lsun_'):len(dataset)-len('256')] + ('_train' if is_train else '_val')
      elif dataset.endswith('128'):
         nm = dataset[len('lsun_'):len(dataset)-len('128')] + ('_train' if is_train else '_val')   
      if nm is None:
         raise ValueError('Unknown dataset set: %s ...' % dataset)
      else:
         if nm.startswith('brlr'):
            indexes = list(range(1300000)) if is_train else list(range(1300000,1315802))
            return gen_lsun_balanced(dataroot, ['bedroom_train', 'living_room_train'], tr, indexes)
         elif nm.startswith('twbg'):
            indexes = list(range(700000)) if is_train else list(range(700000,708264))
            return gen_lsun_balanced(dataroot, ['tower_train', 'bridge_train'], tr, indexes)
         else:
            timeLog('Loading LSUN %s ...' % nm)
            return datasets.LSUN(dataroot, classes=[ nm ], transform=tr)
   else:
      raise ValueError('Unknown dataset: %s ...' % dataset)

#----------------------------------------------------------
def to_pm1(input):
   return input*2-1

#----------------------------------------------------------
def get_tr(dataset, is_train, do_augment):
   if dataset == 'ImageNet':  
      tr = T.Compose([ T.Resize(256), T.CenterCrop(224) ])
   elif dataset == 'ImageNet64':  
      tr = T.Compose([ T.Resize(64), T.CenterCrop(64) ])  
   elif dataset == 'MNIST':
      tr = T.Compose([ T.Pad(2) ]) # 28x28 -> 32x32
   elif dataset == 'FashionMNIST':
      tr = T.Compose([ T.Pad(2) ]) # 28x28 -> 32x32
   elif dataset == 'EMNIST':
      tr = T.Compose([ T.Pad(2) ]) # 28x28 -> 32x32   
   elif dataset.endswith('64'):
      tr = T.Compose([ T.Resize(64), T.CenterCrop(64) ])
   elif dataset.endswith('256'):
      tr = T.Compose([ T.Resize(256), T.CenterCrop(256) ])
   elif dataset.endswith('128'):
      tr = T.Compose([ T.Resize(128), T.CenterCrop(128) ]) 
   elif dataset.endswith('1024'):
      tr = T.Compose([ T.Resize(1024), T.CenterCrop(1024) ])     
   else:
     tr = T.Compose([ ])
     if do_augment:
        tr = T.Compose([
              tr, 
              T.Pad(4, padding_mode='reflect'),
              T.RandomHorizontalFlip(),
              T.RandomCrop(32),
        ])

   return T.Compose([ tr, T.ToTensor(), to_pm1 ])    
         
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
from tqdm import tqdm, trange
def make_dataset(dir, class_to_idx):
  images = []
  dir = os.path.expanduser(dir)
  for target in tqdm(sorted(os.listdir(dir))):
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
      continue

    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        if is_image_file(fname):
          path = os.path.join(root, fname)
          item = (path, class_to_idx[target])
          images.append(item)

  return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)


def default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    return accimage_loader(path)
  else:
    return pil_loader(path)

# class ImageFolder(IterableDataset):
#     def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
#                  repeat=1, cache='none',transform=None):
#         self.repeat = repeat
#         self.cache = cache
#         self.tr= transform
#         if split_file is None:
#             filenames = sorted(os.listdir(root_path))
#         else:
#             with open(split_file, 'r') as f:
#                 filenames = json.load(f)[split_key]
#         if first_k is not None:
#             filenames = filenames[:first_k]

#         self.files = []
#         for filename in filenames:
#             file = os.path.join(root_path, filename)

#             if cache == 'none':
#                 self.files.append(Image.open(file).convert('RGB'))

#     def __len__(self):
#       #   print(len(self.files) * self.repeat)
#         return len(self.files) * self.repeat

#     def __iter__(self):
#         if self.cache == 'none':
#             # return transforms.ToTensor()(Image.open(x).convert('RGB'))
#             return iter(self.tr(self.files))

class ImageFolder_ImageNet(Dataset):
  """A generic data loader where the images are arranged in this way: ::

      root/dogball/xxx.png
      root/dogball/xxy.png
      root/dogball/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

  Args:
      root (string): Root directory path.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      loader (callable, optional): A function to load an image given its path.

   Attributes:
      classes (list): List of the class names.
      class_to_idx (dict): Dict with items (class_name, class_index).
      imgs (list): List of (image path, class_index) tuples
  """

  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, load_in_mem=False, 
               index_filename='imagenet_imgs.npz', **kwargs):
    classes, class_to_idx = find_classes(root)
    # Load pre-computed image directory walk
    if os.path.exists(index_filename):
      print('Loading pre-saved Index file %s...' % index_filename)
      imgs = np.load(index_filename)['imgs']
    # If first time, walk the folder directory and save the 
    # results to a pre-computed file.
    else:
      print('Generating  Index file %s...' % index_filename)
      imgs = make_dataset(root, class_to_idx)
      np.savez_compressed(index_filename, **{'imgs' : imgs})
   #  print('len(imgs){}'.format(len(imgs)))
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                           "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.load_in_mem = load_in_mem
    if self.load_in_mem:
      print('Loading all images into memory...')
      self.data, self.labels = [], []
      for index in tqdm(range(len(self.imgs))):
        path, target = imgs[index][0], imgs[index][1]
        self.data.append(self.transform(self.loader(path)))
        self.labels.append(target)
          

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    if self.load_in_mem:
        img = self.data[index]
        target = self.labels[index]
    else:
      path, target = self.imgs[index]
      img = self.loader(str(path))
      if self.transform is not None:
        img = self.transform(img)
    
    if self.target_transform is not None:
      target = self.target_transform(target)
    
    # print(img.size(), target)
    return img, int(target)

  def __len__(self):
    return len(self.imgs)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str
        

class ImageFolder(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none',transform=None):
        self.repeat = repeat
        self.cache = cache
        self.tr= transform
        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(self.tr(Image.open(file).convert('RGB')))
               #  self.files.append(transforms.ToTensor()(
               #      Image.open(file).convert('RGB')))

    def __len__(self):
      #   print(len(self.files) * self.repeat)
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
      #   print(idx)
        if self.cache == 'none':
            # return transforms.ToTensor()(Image.open(x).convert('RGB'))
            return (self.tr(Image.open(x).convert('RGB')),0)

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x
