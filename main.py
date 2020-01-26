import argparse
import torch
import tqdm

from utils import generate_data, draw_graph, plot_loss
from model import MLP


def main(args):
    x = np.linspace(-6, 6, 100).reshape(100, 1)  # Test data for regression
    x_set, y_set = generate_data()  # Train data for regression
    epochs = args.epochs
    batch_size = args.batch_size
    epsilon = 0.01  # Coeffiecient for 'fast gredient sign method'

    num_hidden = 256
    num_layer = 1
    x_tensor = torch.cuda.LongTensor(x_set)
    y_tensor = torch.cuda.LongTensor(y_set)

    # Training an ensemble of 5 networks(MLPS) with MSE
    # TODO:Draw Fig1.1
    if args.fig == 1:
        means = []
        s = []
        for _ in range(5):
            model = MLP(x_tensor.size(1), num_hidden, num_layer)
            model.cuda()

            loss_fn = torch.nn.MSE()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            train_losses = []
            for _ in tqdm(range(args.epochs), desc='fig_1 training...'):
                total_train_loss = 0

                optimizer.zero_grad()
                y_pred = model.forward(x_tensor)
                loss = loss_fn(y_pred, y_tensor)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                train_losses.append(total_train_loss)

        # Have to calculte predicted mean and std
        # 'mean' have to be a numpy array with shape [100,1]
        # 'std'  have to be a numpy array with shape [100,1]
        mean = np.random.randn(100,1)
        std = np.random.randn(100,1)
        draw_graph(x,x_set,y_set, mean,std)

    # Training a Gaussian MLP(single network) with NLL score rule
    # TODO:Draw Fig1.2
    elif args.fig == 2:

        # Have to calculte predicted mean and var
        # 'mean' have to be a numpy array with shape [100,1]
        # 'var'  have to be a numpy array with shape [100,1]
        mean = np.random.randn(100,1)
        var = np.random.randn(100,1)
        draw_graph(x,x_set,y_set,mean, np.sqrt(var))

    # Training a Gaussian MLP with NLL & Adversarial Training
    # TODO:Draw Fig1.3
    elif args.fig == 3:

        # Have to calculte predicted mean and var
        # 'mean' have to be a numpy array with shape [100,1]
        # 'var'  have to be a numpy array with shape [100,1]
        mean = np.random.randn(100,1)
        var = np.random.randn(100,1)
        draw_graph(x,x_set,y_set,mean, np.sqrt(var))

    # Training a Gaussian mixture MLP (Deep ensemble) with NLL
    # TODO: Draw Fig1.4
    else: #args.fig == 4

        # Have to calculte predicted mean and var
        # 'mean' have to be a numpy array with shape [100,1]
        # 'var'  have to be a numpy array with shape [100,1]
        mean = np.random.randn(100,1)
        var = np.random.randn(100,1)
        draw_graph(x,x_set,y_set,mean, np.sqrt(var))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep ensemble')
    parser.add_argument(
        '--epochs',
        type=int,
        default=300)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20)

    parser.add_argument(
        '--fig',
        type=int)
    args = parser.parse_args()

    main(args)
