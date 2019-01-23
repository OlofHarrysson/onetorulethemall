import visdom
import numpy as np


# Replay weight dominance
# Plot prediction certanty or something?


class Logger():
  def __init__(self):
    self.viz = visdom.Visdom()
    self.mode = 'train'

  def set_logger_mode(self, mode):
    self.mode = mode

  def save_average_loss(self, data, step):
    loss = [d.item() for d in data]
    loss = sum(loss)/len(loss)
    loss = np.array(loss).reshape((1,1))
    title = 'Average Loss ' + self.mode
    opts = dict(title=title)
    self.viz.line(X=[step], Y=loss, update='append', win=title, opts=opts)


  def save_loss_stacked(self, data, step):
    title = 'Percentage Loss ' + self.mode

    data = [d.item() for d in data]
    losses = np.array(data)
    losses_percent = losses / losses.sum()

    tot = 0
    for i, loss in enumerate(losses_percent):
      tot += loss
      losses_percent[i] = tot

    losses_percent = losses_percent.reshape((-1, 1)).T

    win = self.viz.line(
      Y=losses_percent,
      X=[step],
      update='append',
      win=title,
      opts=dict(
          fillarea=True,
          xlabel='Steps',
          ylabel='Percentage',
          title=title,
          stackgroup='one',
      )
    )

  def log_accuracy(self, data, step):
    title = 'Accuracy ' + self.mode
    opts = dict(title=title)
    self.viz.line(X=[step], Y=[data], update='append', win=title, opts=opts)


  def log_ship_accuracy(self, data, step):
    data = data.mean()
    title = 'Ship Accuracy ' + self.mode
    opts = dict(title=title)
    self.viz.line(X=[step], Y=[data], update='append', win=title, opts=opts)

  def log_test_accuracies(self, acc, w_acc, epoch):
    title = 'Average Test Accuracies'
    layoutopts = {'plotly': {'legend': {'x':0, 'y':1}}}
    opts = dict(title=title, legend=['Last Layer Accuracy', 'Weighted Accuracy'], layoutopts=layoutopts)
    data = np.array([acc, w_acc]).reshape(1, -1)
    self.viz.line(X=[epoch], Y=data, update='append', win=title, opts=opts)


  def log_ship_comp(self, w_acc, last_acc, step):
    title = 'Ship Accuracies'
    layoutopts = {'plotly': {'legend': {'x':0, 'y':1}}}
    opts = dict(title=title, legend=['Last Layer Accuracy', 'Weighted Accuracy'], layoutopts=layoutopts)
    data = np.array([last_acc, w_acc]).reshape(1, -1)
    self.viz.line(X=[step], Y=data, update='append', win=title, opts=opts)



  def log_accuracy_per_layer(self, accuracies, step):
    accuracies = np.array(accuracies).reshape(-1, 1).T

    title = 'Accuracy per Layer ' + self.mode
    opts = dict(title=title)
    self.viz.line(X=[step], Y=accuracies, update='append', win=title, opts=opts)


  def max_conf_per_layer(self, confs, step):
    confs = np.array(confs).reshape(-1, 1).T

    title = 'Max Confidence per Layer ' + self.mode
    opts = dict(title=title)
    self.viz.line(X=[step], Y=confs, update='append', win=title, opts=opts)


  def log_heatmap(self, data, classes):
    rownames = ['2nd', '3rd', 'Last']
    opts = dict(title='Prediction Weights', columnnames=list(classes), rownames=rownames, layoutopts={'plotly': {'legend': {'x':0, 'y':0}}})
    self.viz.heatmap(X=data, win='weights', opts=opts)


  def log_accuracy_per_class(self, cls_correct, cls_tot, classes):
    # Accuracy / class
    a, b = cls_correct.astype(np.float32), cls_tot
    with np.errstate(divide='ignore', invalid='ignore'):
      c = np.true_divide(a,b)
      c[c == np.inf] = 0
      c = np.nan_to_num(c)

    rownames = ['2nd', '3rd', 'Last']
    title = 'Accuracy per Class'
    opts = dict(title=title, columnnames=list(classes), rownames=rownames, layoutopts={'plotly': {'legend': {'x':0, 'y':0}}})
    self.viz.heatmap(X=c, win=title, opts=opts)


  def log_prediction_per_class(self, cls_pred, classes):
    rownames = ['2nd', '3rd', 'Last']
    title = 'Prediction Per Class'
    opts = dict(title=title, columnnames=list(classes), rownames=rownames, layoutopts={'plotly': {'legend': {'x':0, 'y':0}}})
    self.viz.heatmap(X=cls_pred, win=title, opts=opts)


class BaselineLogger():
  def __init__(self):
    self.viz = visdom.Visdom()

  def log_accuracy(self, data, step):
    opts = dict(title='Accuracy')
    self.viz.line(X=[step], Y=[data], update='append', win='Accuracy', opts=opts)

  def log_loss(self, data, step):
    opts = dict(title='Loss')
    self.viz.line(X=[step], Y=[data], update='append', win='Loss', opts=opts)