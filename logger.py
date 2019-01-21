import visdom
import numpy as np


# Could be useful to plot the loss in the different layers as training progress. If one layer is way worse than
# Plot weight dominance by layer. Preferably so we can replay it, do this in stacked area graph as we do losses in yolov3
# Plot accuracy
# Plot prediction certanty or something?
# Heatmap of weights


class Logger():
  def __init__(self):
    self.viz = visdom.Visdom()

  def save_average_loss(self, data, step):
    loss = [d.item() for d in data]
    loss = sum(loss)/len(loss)
    loss = np.array(loss).reshape((1,1))
    opts = dict(title='Average Loss')
    self.viz.line(X=[step], Y=loss, update='append', win='Average Loss', opts=opts)


  def save_loss_stacked(self, data, step):
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
      update='append', win='Loss_stacked',
      opts=dict(
          fillarea=True,
          xlabel='Steps',
          ylabel='Percentage',
          title='Percentage Loss',
          stackgroup='one',
      )
    )

  def log_accuracy(self, data, step):
    opts = dict(title='Accuracy')
    self.viz.line(X=[step], Y=[data], update='append', win='Accuracy', opts=opts)


  def log_heatmap(self, data, classes):

    rownames = ['1st', '2nd', '3rd', 'Last']
    opts = dict(title='Prediction Weights', columnnames=list(classes), rownames=rownames, layoutopts={'plotly': {'legend': {'x':0, 'y':0}}})
    self.viz.heatmap(X=data, win='weights', opts=opts)




class BaselineLogger():
  def __init__(self):
    self.viz = visdom.Visdom()

  def log_accuracy(self, data, step):
    opts = dict(title='Accuracy')
    self.viz.line(X=[step], Y=[data], update='append', win='Accuracy', opts=opts)

  def log_loss(self, data, step):
    opts = dict(title='Loss')
    self.viz.line(X=[step], Y=[data], update='append', win='Loss', opts=opts)