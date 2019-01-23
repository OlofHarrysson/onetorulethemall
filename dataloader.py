import torch
import torchvision
from torchvision import transforms

class Dataloader():
  def __init__(self):
    transform = transforms.Compose([
      transforms.ToTensor()
    ])


    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, transform=transform)
    
    self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)

  def get_loader(self):
    return self.trainloader