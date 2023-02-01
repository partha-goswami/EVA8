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
from train import *
from test import *
from utils import *

def experiment(config_dict, train_loader, test_loader):
  '''
  The method performs the experiment as per our configuration.
  To preserve gpu usage, we are exiting while we reach the target validation accuracy.
  '''

  train_losses = []
  test_losses = []
  train_accuracy = []
  test_accuracy = []

  device = get_device()
  model = Net(config_dict['dropout_rate']).to(device)
  scheduler, optimizer = get_scheduler(train_loader, config_dict, model)

  for epoch in range(1, config_dict['epochs'] + 1):
    print(f'Epoch {epoch}:')
    train(model, device, train_loader, optimizer,epoch, train_accuracy, train_losses, config_dict['L1_factor'],
          scheduler,nn.CrossEntropyLoss(),config_dict['gradient_clip'])

    test_accuracy_local = test(model, device, test_loader,test_accuracy,test_losses,nn.CrossEntropyLoss())

    if test_accuracy_local >= config_dict['target_test_accuracy']:
      return (train_accuracy,train_losses,test_accuracy,test_losses)

  return (train_accuracy,train_losses,test_accuracy,test_losses)

def plot_metrics(experiment_results):

  '''
  This method plots Validation Loss & Accuracy progression.
  :param experiment_results: Experiment Results interms of training and testing loss and accuracy
  '''
  sns.set(font_scale=1)
  plt.rcParams["figure.figsize"] = (25,6)
  train_accuracy,train_losses,test_accuracy,test_losses  = experiment_results[0], experiment_results[1], experiment_results[2], experiment_results[3]
  
  # Plot the learning curve.
  fig, (ax1,ax2) = plt.subplots(1,2)
  ax1.plot(np.array(test_losses), 'b', label="Validation Loss")
  
  # Label the plot.
  ax1.set_title("Validation Loss")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Loss")
  ax1.legend()
  
  ax2.plot(np.array(test_accuracy), 'b', label="Validation Accuracy")
  
  # Label the plot.
  ax2.set_title("Validation Accuracy")
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel("Accuracy")
  ax2.legend()
