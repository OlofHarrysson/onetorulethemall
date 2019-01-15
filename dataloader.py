import torch
import torchvision

class Dataloader():
  def __init__(self):
    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True)
    
    self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

  def get_loader(self):
    return self.trainloader