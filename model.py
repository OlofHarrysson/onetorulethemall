from resnet import resnet18
import torch.nn as nn
import numpy as np

class Mymodel(nn.Module):
  def __init__(self):
    super(Mymodel, self).__init__()
    self.resnet18 = resnet18(num_classes=10)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.softmax = nn.Softmax()

    fc1 = nn.Linear(64, 10)
    fc2 = nn.Linear(128, 10)
    fc3 = nn.Linear(256, 10)
    fc4 = nn.Linear(512, 10)

    self.fcs = [fc1, fc2, fc3, fc4]

  def predict(self, preds, labels):
    acc = []
    for pred in preds:
      pred = self.softmax(pred)
      values, indices = pred.max(1)

      correct = (indices == labels).squeeze()
      correct = np.array(correct)
      acc.append(np.sum(correct) / correct.size)
    
    acc_string = ' - '.join(map(str, acc))
    print('Accuracy per layer: {}'.format(acc_string))


  def forward(self, x):
    fmaps = self.resnet18.extract_features(x)

    class_preds = []
    for x, fc in zip(fmaps, self.fcs):
      x = self.avgpool(x)
      x = x.view(x.size(0), -1)
      x = fc(x)
      class_preds.append(x)

    return class_preds


  def calc_loss(self, preds, labels):
    ce_loss = nn.CrossEntropyLoss()
    
    losses = []
    for pred in preds:
      losses.append(ce_loss(pred, labels))

    return losses


class Baseline(nn.Module):
  def __init__(self):
    super(Baseline, self).__init__()
    self.resnet18 = resnet18(num_classes=10)
    self.softmax = nn.Softmax()

  def forward(self, x):
    return self.resnet18(x)

  def predict(self, pred, labels):
    pred = self.softmax(pred)
    values, indices = pred.max(1)

    correct = (indices == labels).squeeze()
    correct = np.array(correct)
    acc = np.sum(correct) / correct.size
    
    print('Accuracy: {}'.format(acc))

  def calc_loss(self, pred, labels):
    ce_loss = nn.CrossEntropyLoss()
    return [ce_loss(pred, labels)]
