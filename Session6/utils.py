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
