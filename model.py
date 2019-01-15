import torchvision.models as models

class Mymodel():
  def __init__(self):
    self.resnet18 = models.resnet18(num_classes=10)



  def forward(self, x):
    fmaps = self.resnet18.extract_features(x)