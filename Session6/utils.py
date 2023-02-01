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


def apply_albumentation(config_dict):
  '''
  Kept separate method to apply albumentation. As it returns the transformation applied in terms of dictionary.
  Our next dataset and dataloaders would use the transformation by calling this method.
  '''
  cifar10_mean,cifar10_std = get_cifar10_stats()

  train_transforms = albumentations.Compose([albumentations.HorizontalFlip(p=config_dict['albumentation']['horizontalFlip_probability']),
                                  albumentations.ShiftScaleRotate(shift_limit=config_dict['albumentation']['shiftScaleRotate_shift_limit'], 
                                                                  scale_limit=config_dict['albumentation']['shiftScaleRotate_scale_limit'],
                                                                  rotate_limit=config_dict['albumentation']['shiftScaleRotate_rotate_limit'],
                                                                  p=config_dict['albumentation']['shiftScaleRotate_probability']),
                                  albumentations.CoarseDropout(max_holes=config_dict['albumentation']['coarseDropout_max_holes'],
                                                               min_holes =config_dict['albumentation']['coarseDropout_min_holes'], 
                                                               max_height=config_dict['albumentation']['coarseDropout_max_height'], 
                                                               max_width=config_dict['albumentation']['coarseDropout_max_width'], 
                                  p=config_dict['albumentation']['coarseDropout_cutout_probability'],fill_value=tuple([x * 255.0 for x in cifar10_mean]),
                                  min_height=config_dict['albumentation']['coarseDropout_min_height'], min_width=config_dict['albumentation']['coarseDropout_min_width']),
                                  albumentations.ColorJitter(p=config_dict['albumentation']['colorJitter_probability'],
                                                             brightness=config_dict['albumentation']['colorJitter_brightness'], 
                                                             contrast=config_dict['albumentation']['colorJitter_contrast'], 
                                                             saturation=config_dict['albumentation']['colorJitter_saturation'], hue=config_dict['albumentation']['colorJitter_hue']),
                                  albumentations.ToGray(p=config_dict['albumentation']['gray_probability']),
                                  albumentations.Normalize(mean=cifar10_mean, std=cifar10_std,always_apply=True),
                                  ToTensorV2()
                                ])

  test_transforms = albumentations.Compose([albumentations.Normalize(mean=cifar10_mean, std=cifar10_std, always_apply=True),
                                 ToTensorV2()])
  return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]

def get_data_loaders(config_dict):
  '''
  This method applies albumentation transforms and returns the train and test dataloaders
  : param config_dict: dictionary of config values
  '''
  train_transforms, test_transforms = apply_albumentation(config_dict)  
  

  trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transforms)  
        
  testset  = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transforms)

  train_loader = torch.utils.data.DataLoader(trainset, 
                                                  batch_size=config_dict['batch_size'], 
                                                  shuffle=True,
                                                  num_workers=config_dict['no_of_workers'], 
                                                  pin_memory=config_dict['pin_memory'])
  test_loader = torch.utils.data.DataLoader(testset, 
                                                 batch_size=config_dict['batch_size'], 
                                                  shuffle=True,
                                                  num_workers=config_dict['no_of_workers'], 
                                                  pin_memory=config_dict['pin_memory'])
  return train_loader, test_loader

def print_model_summary(model, device):
    '''
      THis method returns the model summary.

      :param model: model
      :param device: device
    '''
    cifar_model = model.to(device)
    return summary(cifar_model, input_size=(3, 32, 32))
