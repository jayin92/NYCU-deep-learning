import argparse
import time
import numpy as np

from data import generate_linear, generate_XOR_easy
from utils import show_result, plot_loss
from model import MLP, ReLU, Sigmoid
from loss import MSELoss
from optimizer import SGD


def train(model, loss_fn, optimizer, x, y, epochs=1000):
    losses = []
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_fn.forward(output, y)
        grad = loss_fn.backward()
        model.backward(grad)
        optimizer.step()
        losses.append(loss)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss {loss}')
    print(f'Epoch {epochs}, Loss {loss}')
    print(f'Training finished, elapsed {time.time() - start_time}s')
    # show loss
    if args.output_path:
        plot_loss(
            losses, args, f'{args.output_path}/loss_{args.data}_lr{str(args.lr).replace(".", "")}_n{args.num_neurons}_e{args.epochs}{"_no_acti" if args.no_activation else ""}.png')
    else:
        plot_loss(losses, args)


def test(model, x, y):
    output = model.forward(x)
    binary_output = np.where(output >= 0.5, 1, 0)
    correct = 0
    for i in range(len(y)):
        if y[i] == binary_output[i]:
            correct += 1
    print(f'Accuracy: {correct}/{len(y)} = {correct/len(y)*100}%')
    if args.output_path:
        show_result(x, y, binary_output,
                    f'{args.output_path}/{args.data}_lr{str(args.lr).replace(".", "")}_n{args.num_neurons}_e{args.epochs}{"_no_acti" if args.no_activation else ""}.png')
    else:
        show_result(x, y, binary_output)


def main(args):
    # Generate data
    if args.data == 'linear':
        x, y = generate_linear()
    elif args.data == 'xor':
        x, y = generate_XOR_easy()
    else:
        raise ValueError('Invalid data')
    # Initialize model
    activations = [None, Sigmoid()] if args.no_activation else [ReLU(), Sigmoid()]
    model = MLP(2, args.num_neurons, 1, activations=activations)

    # Initialize loss
    loss_fn = MSELoss()
    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=args.lr)
    # Train
    train(model, loss_fn, optimizer, x, y, epochs=args.epochs)
    # Test
    test(model, x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, choices=['linear', 'xor'], default='xor'
    )
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--num_neurons', type=int, default=16)
    parser.add_argument('--no_activation', action='store_true')
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
