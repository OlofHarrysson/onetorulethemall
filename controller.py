import torch
from utils.utils import ProgressbarWrapper as Pbar

class Controller(object):
  def __init__(self, model, dataloader):
    super(Controller, self).__init__()
    self.model = model
    self.dataloader = dataloader

  def train(self):
    model = self.model
    device = model.device

    # params = model.parameters()
    branch_params = []
    other_params = []

    for name, par in model.named_parameters():
      attr_name = name.split('.')[0]
      if attr_name == 'branches':
        branch_params.append(par)
      else:
        other_params.append(par)

    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(other_params)
    branch_optimizer = torch.optim.Adam(branch_params)

    n_epochs = 10
    n_batches = len(self.dataloader.get_loader())
    pbar = Pbar(n_epochs, n_batches)

    optim_steps = 0
    for epoch in pbar(range(1, n_epochs + 1)):
      for batch_i, data in enumerate(self.dataloader.get_loader(), 1):
        pbar.update(epoch, batch_i)

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        branch_optimizer.zero_grad()

        outputs = model(inputs)
        losses = model.calc_loss(outputs, labels, optim_steps)
        
        sum_loss = sum(losses)
        sum_loss.backward()
        optimizer.step()
        branch_optimizer.step()
        optim_steps += 1

        model.predict(outputs, labels, optim_steps)
        del outputs # Frees up GPU memory
        del sum_loss
