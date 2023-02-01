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
from model import *
from utils import *


def test(model, device, test_loader,test_acc,test_losses,criterion):

  '''
  This method is responsible for model testing
  :param model: model
  :param device: device, cuda (gpu), or cpu
  :param test_loader: test loader
  :param test_acc: test accuracy
  :param test_losses: test losses
  :param criterion: criterion
  '''
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          test_loss += criterion(output, target).item()
          pred = output.argmax(dim=1, keepdim=True)  
          correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
  
  test_acc.append(100. * correct / len(test_loader.dataset))
  return 100. * correct / len(test_loader.dataset)
