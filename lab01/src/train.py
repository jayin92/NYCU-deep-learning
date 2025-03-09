import numpy as np

from data import generate_linear, generate_XOR_easy
from utils import show_result
from model import MLP
from loss import MSELoss
from optimizer import SGD


def train(model, loss_fn, optimizer, x, y, epochs=1000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_fn.forward(output, y)
        grad = loss_fn.backward()
        model.backward(grad)
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss {loss}')
    print(f'Epoch {epochs}, Loss {loss}')
    print('Training finished')


def test(model, x, y):
    output = model.forward(x)
    binary_output = np.where(output >= 0.5, 1, 0)
    correct = 0
    for i in range(len(y)):
        if y[i] == binary_output[i]:
            correct += 1
    print(f'Accuracy: {correct}/{len(y)} = {correct/len(y)*100}%')
    show_result(x, y, binary_output)


def main():
    # Generate data
    x, y = generate_XOR_easy()
    # Initialize model
    model = MLP(2, 32, 1)
    # Initialize loss
    loss_fn = MSELoss()
    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=0.01)
    # Train
    train(model, loss_fn, optimizer, x, y, epochs=100000)
    # Test
    test(model, x, y)


if __name__ == '__main__':
    main()
