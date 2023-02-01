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


def train(model, device, train_loader, optimizer, epoch,train_acc,train_loss,l1_factor,scheduler,criterion,grad_clip=None):
  '''
  This method is responsible for model training

  :param model: model
  :param device: cuda (gpu) or cpu
  :param train_loader: train loader
  :param optimizer: Optimizer, for example, Adam, or SGD
  :param epoch: epoch, the number of times we are seeing the entire training data
  :param train_acc: training accuracy
  :param train_loss: training loss
  :param l1_factor: L1 Factor
  :param scheduler: scheduler
  :param criterion: criterion
  :param grad_clip: gradient clipping value
  '''

  model.train()
  pbar = tqdm(train_loader)
  correct, processed = 0, 0
  
  for batch_idx, (data, target) in enumerate(pbar):

    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    y_pred = model(data)
    loss = criterion(y_pred, target)
    
    if l1_factor > 0:
      l1 = 0
      for p in model.parameters():
        l1 = l1 + p.abs().sum()
      loss = loss + l1_factor*l1

    train_loss.append(loss.data.cpu().numpy().item())
    loss.backward()

    if grad_clip: 
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        
    optimizer.step()
    scheduler.step()
    pred = y_pred.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
