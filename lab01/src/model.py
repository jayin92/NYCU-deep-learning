import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        
        # Convert weights and biases to Parameter objects
        self.weight = Parameter(np.random.randn(out_features, in_features))
        self.bias = Parameter(np.random.randn(out_features)) if bias else None
    
    def forward(self, x):
        self.x = x
        if self.bias is not None:
            return np.dot(x, self.weight.data.T) + self.bias.data
        return np.dot(x, self.weight.data.T)
    
    def backward(self, grad):
        # Store gradients in Parameter objects
        self.weight.grad += np.dot(grad.T, self.x)
        if self.bias is not None:
            self.bias.grad += np.sum(grad, axis=0)
        return np.dot(grad, self.weight.data)
    
    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]


# Activation functions
class Sigmoid:
    def __init__(self):
        self.output = None
        
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1 - self.output)
    
    def parameters(self):
        return []  # Activation functions don't have trainable parameters


class ReLU:
    def __init__(self):
        self.input = None
        
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * np.where(self.input > 0, 1, 0)
    
    def parameters(self):
        return []  # Activation functions don't have trainable parameters


# MLP
class MLP:
    def __init__(self, in_features, hidden_size, out_features):
        self.linear1 = Linear(in_features, hidden_size)
        self.activation1 = ReLU()
        self.linear2 = Linear(hidden_size, out_features)
        self.activation2 = Sigmoid()

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.activation1.forward(x)
        x = self.linear2.forward(x)
        x = self.activation2.forward(x)
        return x
    
    def backward(self, grad):
        grad = self.activation2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.activation1.backward(grad)
        grad = self.linear1.backward(grad)
        return grad

    def parameters(self):
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params
