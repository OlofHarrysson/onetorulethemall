from resnet import resnet18
import torch.nn as nn
import numpy as np
from collections import defaultdict
from logger import Logger

class Mymodel(nn.Module):
  def __init__(self, classes):
    super(Mymodel, self).__init__()
    self.classes = classes
    n_classes = len(classes)
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
    # print(self.pred_weights)


    self.logger = Logger()

  def predict(self, preds, labels, step):
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
    if w_acc >= mean_acc:
      print('Weighted acc is bigger by {}'.format(w_acc - mean_acc))
    else:
      print('nooooop')
    # print('Weighted Acc: {}. Mean Acc: {}'.format(w_acc, mean_acc))

    self.logger.log_accuracy(w_acc, step)

    self.update_pred_weights(numpy_preds, labels)

    self.logger.log_heatmap(self.pred_weights, self.classes)


  def handle_back_prob(self):
    pass
    # I don't want to force each layer to become good at every class. I want them to be able to specialize at one or a few classes. Thereby I can't enforce a huge loss when they fuck up badly like we normally do in back prop.

    # If a layer predicts highly for the wrong class we want to decrease that prediction. But do we want to increase the right class? Then we are forcing every layer to learn everything which we wrote above that we didn't want to. On the other hand, if we never increase the right answer then how could we prevent a sitation that a layer was unlucky in the beginning/filters change and we never give it a chance to recover?

    # If a layer makes a right prediction, with like 0.8 prob we want to increase that to 0.9 - so here we have a loss for that layer+class combo.





  def update_pred_weights(self, preds, labels):
    weights = self.pred_weights

    # The minimum multiplier. Decrease to change weights faster
    discount = 0.99

    labels = labels.numpy()

    # For every layer
    for i, layer_pred in enumerate(preds):
      pred_per_class = defaultdict(lambda: [])

      # Take prediction (0,1) for labels
      for j, label in enumerate(labels):
        # The predicted value for the label
        vv = layer_pred[j][label]
        pred_per_class[label].append(vv)


      # Update weights with pred[label]
      for label, pred_list in pred_per_class.items():
        mean_v = np.mean(np.array(pred_list))

        # Good pred -> high mean_v -> multiplier =ish 1
        mult = discount * (1 - mean_v)
        weights[i][label] *= mult


    self.normalize_weights(weights)

  def normalize_weights(self, weights):
    weights = weights/weights.sum(0)
    self.pred_weights = weights


  def forward(self, x):
    fmaps = self.resnet18.extract_features(x)

    class_preds = []
    for x, fc in zip(fmaps, self.fcs):
      x = self.avgpool(x)
      x = x.view(x.size(0), -1)
      x = fc(x)
      class_preds.append(x)

    return class_preds


  def calc_loss(self, preds, labels, step):
    # TODO: Loss lambdas?
    ce_loss = nn.CrossEntropyLoss()
    
    losses = []
    for loss_lambda, pred in enumerate(preds):
      # losses.append(loss_lambda * ce_loss(pred, labels))
      losses.append(ce_loss(pred, labels))

    loss = losses[0].item()
    loss = np.array(loss).reshape(1, -1)
    self.logger.save_loss(loss, step)

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
