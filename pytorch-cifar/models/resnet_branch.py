'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_copy import ResNet18Copy, ResNet101Copy, ResNet50Copy
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
    self.be_loss = nn.BCELoss()
    self.sigmoid = nn.Sigmoid()

    b2 = Branch(128, n_classes, kern_size=3)
    b3 = Branch(256, n_classes, kern_size=2) # TODO: To small fmap

    self.branches = nn.Sequential(*[b2, b3])

    n_pred_layers = 3 # branches + final
    self.n_pred_layers = n_pred_layers
    self.pred_weights = np.full((n_pred_layers, n_classes), 1/n_pred_layers)
    self.class_correct = np.full((n_pred_layers, n_classes), 0)
    self.class_total = list(0. for i in range(n_classes))

    self.class_predicted = np.full((n_pred_layers, n_classes), 0)

    self.predictions = defaultdict(lambda: [])
    self.all_labels = []

    self.confs = []
    self.correct = []

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

    # TODO: Rewries the weights with accuracies
    # cls_correct, cls_tot = self.class_correct, self.class_total
    # a, b = cls_correct.astype(np.float32), cls_tot
    # with np.errstate(divide='ignore', invalid='ignore'):
    #   c = np.true_divide(a,b)
    #   c[c == np.inf] = 0
    #   c = np.nan_to_num(c)
    # self.pred_weights = c

    self.all_labels.extend(labels.numpy())

    # Add to seen classes
    for i in range(labels.size(0)):
      label = labels[i]
      self.class_total[label] += 1


    acc = []
    w_preds = []
    numpy_preds = []
    max_conf_per_layer = []
    for layer, pred in enumerate(preds):
      pred = pred.cpu()
      
      # Last layer uses softmax preds
      if layer == len(preds) - 1:
        pred = self.softmax(pred)
      else:
        # pred = self.softmax(pred)
        pred = self.sigmoid(pred)

      values, indices = pred.max(1)
      correct = (indices == labels).squeeze()



      # For confusion matrix
      self.predictions[layer].extend(indices.numpy())

      # Classes predicted
      for pred_i in indices:
        self.class_predicted[layer][pred_i] += 1

      # Classes correct
      for i in range(correct.size(0)):
        label = labels[i]
        self.class_correct[layer][label] += correct[i].item()


      correct = np.array(correct)
      acc.append(np.sum(correct) / correct.size)

      pred_tmp = pred.detach().numpy()
      w_pred = pred_tmp * self.pred_weights[layer]
      w_preds.append(w_pred)

      numpy_preds.append(pred_tmp)
      max_pred = pred_tmp.max(axis=1).mean()
      max_conf_per_layer.append(max_pred)

       # Last layer
      if layer == len(preds) - 1:
        self.confs.extend(values.detach().numpy())
        self.correct.extend(correct)


    weighted_pred = self.predict_with_best_layer(numpy_preds)

    # Weighted Acc
    # weighted_pred = sum(w_preds)
    indices = np.argmax(weighted_pred, axis=1)
    w_correct = (indices == labels.numpy()).astype(np.int64)
    w_acc = np.sum(w_correct) / w_correct.size

    self.logger.log_conf_correct_or_not(self.confs, self.correct)
    self.logger.log_accuracy_per_layer(acc, step)
    # self.logger.max_conf_per_layer(max_conf_per_layer, step)
    self.logger.log_accuracy(w_acc, step)
    self.logger.log_heatmap(self.pred_weights, self.classes)
    recall = self.logger.log_accuracy_per_class(self.class_correct, self.class_total, self.classes)
    preci = self.logger.log_precision_per_class(self.class_correct, self.class_predicted, self.classes)
    self.logger.log_prediction_per_class(self.class_predicted, self.classes)

    self.logger.log_f1_per_class(preci, recall, self.classes)

    self.logger.log_confusion_matrices(self.all_labels, self.predictions, self.classes)

    if is_train:
      self.update_pred_weights(numpy_preds, labels)

    return np.sum(w_correct)


  def update_pred_weights(self, preds, labels):
    weights = self.pred_weights

    # The minimum multiplier. Decrease to change weights faster
    discount = 0.99

    labels = labels.numpy()

    # For every layer prediction
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

  def predict_with_best_layer(self, preds):
    # Uses precision instead of acc now
    # What we do now is saying - you layer have the best precision so you are in charge for this class - even with poor recall. With low recall it means that it doesn't predict the class often enough - it's a bit careful with its preds.
    # What we instead want to do is say - hey layer, if you are sure this is a class we trust you due to your good precision. If you however don't think it's your speciallity class then don't bother helping out.

    # We want to prevent last layer to predict cat all the time.
    # Cant trust last layer when it says cat=1 - bad precision. We can trust companion layer when it says cat=1 - just it doesn't do that very often. We can't trust companion layer when it says cat=0 because low recall. We need someone who can say cat=0 and that we trust - someone with high recall. It might not say cat=0 very often - but hopefully often enough to change some 1->0.


    # Cant trust last layer when it says frog=0 - bad recall. Need someone with high precision who can say - this is a frog. Change 0->1.

    a, b = self.class_correct.astype(np.float32), self.class_predicted
    with np.errstate(divide='ignore', invalid='ignore'):
      c = np.true_divide(a,b)
      c[c == np.inf] = 0
      c = np.nan_to_num(c)
    
    layer_acc = c
    last_layer_acc = layer_acc[-1]
    last_preds = preds[-1]

    # Best layer per class
    best_layer_acc = np.argmax(layer_acc, axis=0)

    # Replace all class_preds in batch with best_layers pred
    for cls_i, best_layer in enumerate(best_layer_acc):
      best_preds = preds[best_layer][:, cls_i]
      last_preds[:, cls_i] = best_preds

    return last_preds

  def calc_trunk_loss(self, preds, labels, step):
    # TODO: Loss lambdas?
    # Paper updated layers based on weight * loss. Low weight meant we shouldn't really think about loss to much. Also set a minimum weight of s/#layers. This was done so a layer could recover if bad in start.
  
    loss = self.ce_loss(preds, labels)

    # self.logger.save_average_loss(losses, step)
    # self.logger.save_loss_stacked(losses, step)

    return loss


  def calc_branch_loss(self, preds, labels, step):
    # l1_loss = nn.L1Loss()
    losses = []
    for pred in preds:
      values, indices = pred.max(1)
      values = self.sigmoid(values)
      correct = (indices == labels).type(torch.FloatTensor)
      correct = correct.to('cuda')

      losses.append(self.be_loss(values, correct))
      # losses.append(l1_loss(values, correct))


    # self.logger.save_average_loss(losses, step)
    # self.logger.save_loss_stacked(losses, step)

    return sum(losses)

  def set_logger_mode(self, mode):
    self.logger.set_logger_mode(mode)


  def log_accuracies(self, acc, w_acc, epoch):
    self.logger.log_test_accuracies(acc, w_acc, epoch)

  def reset_class_acc(self):
    n_classes = len(self.classes)
    self.class_correct = np.full((self.n_pred_layers, n_classes), 0)
    self.class_total = list(0. for i in range(n_classes))

    self.class_predicted = np.full((self.n_pred_layers, n_classes), 0)

    self.pred_weights = np.full((self.n_pred_layers, n_classes), 1/self.n_pred_layers)

    self.predictions = defaultdict(lambda: [])
    self.all_labels = []

    self.confs = []
    self.correct = []