from model import Mymodel, Baseline
from controller import Controller
from dataloader import Dataloader

def main():
  dataloader = Dataloader()
  
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  # Create model
  model = Mymodel(classes)
  # model = Baseline()
  controller = Controller(model, dataloader)
  controller.train()

if __name__ == '__main__':
  main()