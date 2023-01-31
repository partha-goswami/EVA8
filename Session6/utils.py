from torchsummary import summary
import torch
import model
import numpy as np

def print_model_summary(model, device):
    '''
      THis method returns the model summary.
      :param model: model
      :param device: device
    '''
    cifar_model = model.to(device)
    return summary(cifar_model, input_size=(3, 32, 32))

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

      'grey_probability': 0.15
  }
  return dict_config_values
