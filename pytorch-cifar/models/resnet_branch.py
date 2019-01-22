'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_copy import ResNet18Copy
import numpy as np
from collections import defaultdict
from logger import Logger

class Branch(nn.Module):
  def __init__(self, in_filters, n_classes, kern_size):
    super(Branch, self).__init__()
    self.conv1 = nn.Conv2d(in_filters, in_filters, kernel_size=kern_size)
    self.fc = nn.Linear(in_filters, n_classes)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # self.branches = nn.Sequential(*[b2, b3])

  def forward(self, x):
    x = self.conv1(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


class ResNet18Branch(nn.Module):
  def __init__(self, classes):
    super(ResNet18Branch, self).__init__()
    self.resnet18 = ResNet18Copy()
    self.softmax = nn.Softmax()
    self.logger = Logger()
    self.classes = classes
    n_classes = len(classes)
    
    self.ce_loss = nn.CrossEntropyLoss()

    b2 = Branch(128, n_classes, kern_size=3)
    b3 = Branch(256, n_classes, kern_size=2) # TODO: To small fmap

    self.branches = nn.Sequential(*[b2, b3])

    n_pred_layers = 3 # branches + final
    self.pred_weights = np.full((n_pred_layers, n_classes), 1/n_pred_layers)


  def forward(self, x):
    fmaps = self.resnet18.forward(x)

    class_preds = []
    # Not last fmap
    assert len(self.branches) == len(fmaps) - 1
    for x, branch in zip(fmaps, self.branches):
      x = branch(x)
      class_preds.append(x)

    # Last fmap
    x = fmaps[-1]
    class_preds.append(x)

    return class_preds


  def w_predict(self, preds, labels, step, is_train=False):
    labels = labels.cpu()

    acc = []
    w_preds = []
    numpy_preds = []
    for layer, pred in enumerate(preds):
      pred = pred.cpu()
      pred = self.softmax(pred)
      values, indices = pred.max(1)
      # acc_layer = indices.eq(labels).sum().item() / pred.size(0)

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
    w_correct = (indices == labels.numpy()).astype(np.int64)
    w_acc = np.sum(w_correct) / w_correct.size


    self.logger.log_accuracy_per_layer(acc, step)
    self.logger.log_accuracy(w_acc, step)
    self.logger.log_heatmap(self.pred_weights, self.classes)

    if is_train:
      self.update_pred_weights(numpy_preds, labels)

    return np.sum(w_correct)


  def update_pred_weights(self, preds, labels):
    weights = self.pred_weights

    # The minimum multiplier. Decrease to change weights faster
    discount = 0.99

    labels = labels.numpy()

    # For every layer softmax prediction
    for i, layer_pred in enumerate(preds):
      pred_per_class = defaultdict(lambda: [])

      # Take prediction (0,1) for labels
      for j, label in enumerate(labels):
        vv = layer_pred[j][label]
        pred_per_class[label].append(vv)


      # Update weights with pred[label]
      for label, pred_list in pred_per_class.items():
        mean_v = np.mean(np.array(pred_list))

        # Good pred -> high mean_v -> multiplier =ish 1
        mult = discount ** (1 - mean_v)
        weights[i][label] *= mult


    self.normalize_weights(weights)

  def normalize_weights(self, weights):
    weights = weights/weights.sum(0)
    self.pred_weights = weights

  def calc_loss(self, preds, labels, step):
    # TODO: Loss lambdas?
    # Paper updated layers based on weight * loss. Low weight meant we shouldn't really think about loss to much. Also set a minimum weight of s/#layers. This was done so a layer could recover if bad in start.
    
    losses = []
    for loss_lambda, pred in enumerate(preds):
      losses.append(self.ce_loss(pred, labels))

    self.logger.save_average_loss(losses, step)
    self.logger.save_loss_stacked(losses, step)

    return sum(losses)

  def set_logger_mode(self, mode):
    self.logger.set_logger_mode(mode)


  def log_accuracies(self, acc, w_acc, epoch):
    self.logger.log_test_accuracies(acc, w_acc, epoch)