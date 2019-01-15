from model import Mymodel
from controller import Controller
from dataloader import Dataloader

def main():
  dataloader = Dataloader()
  
  # Create model
  model = Mymodel()
  controller = Controller(model, dataloader)
  controller.train()

if __name__ == '__main__':
  main()