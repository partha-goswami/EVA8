import torch
import torchvision
from torchvision import datasets, transforms
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,OneCycleLR
import seaborn as sns
import matplotlib.pyplot as plt

def get_cifar10_stats():
  '''
  we are calculating mean and standard deviation for each channel of the input data. We would use this in cutout fillvalue in albumentation transformation
  '''
  train_transform = transforms.Compose([transforms.ToTensor()])
  train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
  return train_set.data.mean(axis=(0,1,2))/255, train_set.data.std(axis=(0,1,2))/255

def get_config_values():
  '''
  Returns the config data used
  '''

  dict_config_values = {}
  dict_config_values['dropout_rate'] = 0.01
  dict_config_values['batch_size'] = 256
  dict_config_values['no_of_workers'] = 2
  dict_config_values['pin_memory'] = True
  dict_config_values['learning_rate'] = 0.01
  dict_config_values['epochs'] = 90
  dict_config_values['L1_factor'] = 0
  dict_config_values['L2_factor'] = 0.0001
  dict_config_values['gradient_clip'] = 0.1
  dict_config_values['target_test_accuracy'] = 85

  
  dict_config_values['albumentation'] = {
      'horizontalFlip_probability': 0.2,

      'shiftScaleRotate_shift_limit': 0.1,
      'shiftScaleRotate_scale_limit': 0.1,
      'shiftScaleRotate_rotate_limit': 15,
      'shiftScaleRotate_probability': 0.25,

      'coarseDropout_max_holes': 1,
      'coarseDropout_min_holes': 1,
      'coarseDropout_max_height': 16,
      'coarseDropout_max_width': 16,
      'coarseDropout_min_height': 16,
      'coarseDropout_min_width': 16,
      'coarseDropout_cutout_probability': 0.5,

      'colorJitter_probability': 0.25,
      'colorJitter_brightness': 0.3,
      'colorJitter_contrast': 0.3,
      'colorJitter_saturation': 0.3,
      'colorJitter_hue': 0.2,

      'gray_probability': 0.15
  }
  return dict_config_values
