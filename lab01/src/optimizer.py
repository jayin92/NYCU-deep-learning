import numpy as np

class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad
            param.grad = np.zeros_like(param.grad)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super(SGD, self).__init__(parameters, lr)

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad
            param.grad = np.zeros_like(param.grad)
