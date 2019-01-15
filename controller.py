import torch
from utils.utils import ProgressbarWrapper as Pbar

class Controller(object):
  def __init__(self, model, dataloader):
    super(Controller, self).__init__()
    self.model = model
    self.dataloader = dataloader

  def train(self):
    model = self.model
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    optimizer = torch.optim.Adam(model.parameters())

    n_epochs = 3
    n_batches = len(self.dataloader.get_loader())
    pbar = Pbar(n_epochs, n_batches)

    optim_steps = 0
    for epoch in pbar(range(1, n_epochs + 1)):
      for batch_i, data in enumerate(self.dataloader.get_loader(), 1):
        pbar.update(epoch, batch_i)

        inputs, labels, _ = data
        image_size = inputs.size(2)

        print(image_size)

        # optimizer.zero_grad()
        # outputs = model(inputs)

        # losses = model.calc_loss(outputs, labels, image_size)
        # sum_loss = sum(losses.values())
        # sum_loss.backward()
        # optimizer.step()
        # optim_steps += 1

        del outputs # Frees up GPU memory
        del sum_loss
