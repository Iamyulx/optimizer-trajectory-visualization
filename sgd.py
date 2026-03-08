class SGD:

  def __init__(self, lr=0.001):
    self.lr = lr # Initialize lr attribute

  def step(self, x, grad):
    return x - self.lr * grad # Add the step method for optimization
