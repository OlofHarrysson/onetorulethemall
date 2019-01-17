from resnet import resnet18
import torch.nn as nn
import numpy as np

class Mymodel(nn.Module):
  def __init__(self):
    super(Mymodel, self).__init__()
    self.resnet18 = resnet18(num_classes=10)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    fc1 = nn.Linear(64, 10)
    fc2 = nn.Linear(128, 10)
    fc3 = nn.Linear(256, 10)
    fc4 = nn.Linear(512, 10)

    self.fcs = [fc1, fc2, fc3, fc4]


  def forward(self, x):
    fmaps = self.resnet18.extract_features(x)

    class_preds = np.array((4,10))
    for x, fc in zip(fmaps, self.fcs):
      x = self.avgpool(x)
      x = x.view(x.size(0), -1)
      x = fc(x)
      np.app
      class_preds.append(x)


    class_preds = np.array(class_preds)
    print(class_preds.shape)