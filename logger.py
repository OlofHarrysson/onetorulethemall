import visdom

# Could be useful to plot the loss in the different layers as training progress. If one layer is way worse than
# Plot weight dominance by layer. Preferably so we can replay it, do this in stacked area graph as we do losses in yolov3
# Plot accuracy
# Plot prediction certanty or something?
# Heatmap of weights


class Logger():
  def __init__(self):
    self.viz = visdom.Visdom()

  def save_loss(self, data, step):
    opts = dict(title='Loss')
    self.viz.line(X=[step], Y=data, update='append', win='Loss', opts=opts)

  def log_accuracy(self, data, step):
    opts = dict(title='Accuracy')
    self.viz.line(X=[step], Y=[data], update='append', win='Accuracy', opts=opts)


  def log_heatmap(self, data, classes):
    rownames = ['1st', '2nd', '3rd', 'Last']
    opts = dict(title='Prediction Weights', columnnames=list(classes), rownames=rownames)
    self.viz.heatmap(X=data, win='weights', opts=opts)