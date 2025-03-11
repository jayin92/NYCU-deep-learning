import matplotlib.pyplot as plt


def show_result(x, y, pred_y, output_path=None):    
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    if output_path:
        plt.savefig(output_path)

    plt.show()
    


def plot_loss(losses, args, output_path=None):
    plt.title(f'Training loss with lr={args.lr}, neurons={args.num_neurons}, epochs={args.epochs} {"No activation" if args.no_activation else ""}')
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if output_path:
        plt.savefig(output_path)

    plt.show()
    