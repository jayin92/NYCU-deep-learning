import numpy as np


class Loss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        pass

    def backward(self):
        pass


class MSELoss(Loss):    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / len(self.y_true)
    