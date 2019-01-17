from resnet import resnet18
import torch.nn as nn
import numpy as np

class Mymodel(nn.Module):
  def __init__(self):
    super(Mymodel, self).__init__()
    n_classes = 10
    self.resnet18 = resnet18(num_classes=n_classes)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.softmax = nn.Softmax()

    fc1 = nn.Linear(64, 10)
    fc2 = nn.Linear(128, 10)
    fc3 = nn.Linear(256, 10)
    fc4 = nn.Linear(512, 10)

    self.fcs = [fc1, fc2, fc3, fc4]

    n_pred_layers = 4
    self.pred_weights = np.full((n_pred_layers, n_classes), 1/4)
    print(self.pred_weights)

  def predict(self, preds, labels):
    acc = []
    w_preds = []
    numpy_preds = []
    for layer, pred in enumerate(preds):
      pred = self.softmax(pred)
      values, indices = pred.max(1)

      correct = (indices == labels).squeeze()
      correct = np.array(correct)
      acc.append(np.sum(correct) / correct.size)

      pred_tmp = pred.detach().numpy()
      w_pred = pred_tmp * self.pred_weights[layer]
      w_preds.append(w_pred)

      numpy_preds.append(pred_tmp)

    # Weighted Acc
    weighted_pred = sum(w_preds)
    indices = np.argmax(weighted_pred, axis=1)
    correct = (indices == labels.numpy()).astype(np.int64)
    w_acc = np.sum(correct) / correct.size
    
    # acc_string = ' - '.join(map(str, acc))
    # print('Accuracy per layer: {}'.format(acc_string))

    mean_acc = np.mean(acc)
    print('Weighted Acc: {}. Mean Acc: {}'.format(w_acc, mean_acc))


    self.update_pred_weights(numpy_preds, labels)


  def update_pred_weights(self, preds, labels):
    weights = self.pred_weights

    discount = 0.95

    labels = labels.numpy()

    for i, layer_pred in enumerate(preds):
      # The predicted value for the label
      layer_pred_v = layer_pred[np.arange(len(layer_pred)), labels]
      layer_pred_v = np.mean()

      mult = discount * (1 - layer_pred_v)

      # Update weight matrix per class. Cant do mean because several classes in there
      weights = weights[]



    qwe



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
