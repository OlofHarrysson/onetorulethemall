import visdom


class Logger():
  def __init__(self):
    self.vis = visdom.Visdom()

  def save_loss(self, data):
    self.vis.line(Y=data)

