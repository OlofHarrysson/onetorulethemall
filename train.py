from model import Mymodel, Baseline
from controller import Controller
from dataloader import Dataloader
from logger import Logger, BaselineLogger
import torch


# Why this might just work. Normal NN -> Each layer reads the state from its preceding layer and writes to the subsequent layer. It changes the state but also passes on information that needs to be preserved.

# This might also explain why we want more filters later on since we need a few filters to pass the signal along and a few to combine them.

def main():
  dataloader = Dataloader()
  
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  device = 'cuda' if torch.cuda.is_available() else 'cpu'


  # Create model
  logger = Logger()
  model = Mymodel(classes, logger, device)

  # logger = BaselineLogger()
  # model = Baseline(logger, device)

  controller = Controller(model, dataloader)
  controller.train()

if __name__ == '__main__':
  main()