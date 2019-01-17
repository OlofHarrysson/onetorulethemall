from model import Mymodel, Baseline
from controller import Controller
from dataloader import Dataloader

def main():
  dataloader = Dataloader()
  
  # Create model
  # model = Mymodel()
  model = Baseline()
  controller = Controller(model, dataloader)
  controller.train()

if __name__ == '__main__':
  main()