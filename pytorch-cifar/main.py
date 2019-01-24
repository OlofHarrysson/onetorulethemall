'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.resnet import ResNet18
from models.resnet_branch import ResNet18Branch
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18Branch(classes)
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/ckpt.t7')
    checkpoint = torch.load('./checkpoint/trunk_only_85per_val.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


branch_params = []
other_params = []
for name, par in net.named_parameters():
  attr_name = name.split('.')[0]
  if attr_name == 'branches':
    branch_params.append(par)
  else:
    other_params.append(par)


# Train main "only"
optimizer = optim.SGD(other_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
branch_optimizer = optim.SGD(branch_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Train all layers
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# branch_optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch, optim_steps):
  print('\nEpoch: %d' % epoch)
  net.train()
  net.set_logger_mode('train')
  net.reset_class_acc()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    branch_optimizer.zero_grad()
    outputs = net(inputs)
    trunk_loss = net.calc_trunk_loss(outputs[-1], targets, optim_steps)
    # branch_loss = net.calc_branch_loss(outputs[:-1], targets, optim_steps)

    trunk_loss.backward()
    optimizer.step()
    
    # branch_loss.backward()
    # branch_optimizer.step()

    train_loss += trunk_loss.item()
    last_output = outputs[-1]
    _, predicted = last_output.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    net.w_predict(outputs, targets, optim_steps, is_train=True)
    optim_steps += 1

    # if optim_steps % 10 == 0:
    #   break

    del trunk_loss
    # del branch_loss
    del outputs
    
  return optim_steps

def test(epoch, optim_steps):
  global best_acc
  net.eval()
  net.set_logger_mode('val')
  net.reset_class_acc()
  test_loss = 0
  correct = 0
  total = 0
  w_correct = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      trunk_loss = net.calc_trunk_loss(outputs[-1], targets, optim_steps)

      test_loss += trunk_loss.item()
      last_output = outputs[-1]
      _, predicted = last_output.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

      w_correct += net.w_predict(outputs, targets, optim_steps)
      optim_steps += 1

    # Save checkpoint.

    acc = correct/total
    w_acc = w_correct/total

    net.log_accuracies(acc, w_acc, epoch)

    acc *= 100. # They do it in percent
    if acc > best_acc:
      print('Saving..')
      state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
      }
      if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
      torch.save(state, './checkpoint/ckpt.t7')
      best_acc = acc

  return optim_steps

optim_steps = 0
test_optim_steps = 0
for epoch in range(start_epoch, start_epoch+200):
  optim_steps = train(epoch, optim_steps)
  test_optim_steps = test(epoch, test_optim_steps)
