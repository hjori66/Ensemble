import argparse
import torch
import numpy as np

from utils import generate_data, draw_graph, plot_loss
from model import MLP, GaussianMLP
from tqdm import tqdm

np.random.seed(1)
torch.cuda.manual_seed(1)
torch.manual_seed(1)

def main(args):
    x = np.linspace(-6, 6, 100).reshape(100, 1)  # Test data for regression
    x_set, y_set = generate_data()  # Train data for regression
    epochs = args.epochs
    batch_size = args.batch_size
    epsilon = 0.01  # Coeffiecient for 'fast gredient sign method'

    num_hidden = 100
    num_layer = 1
    is_cuda = False

    if is_cuda:
        x_line_tensor = torch.cuda.FloatTensor(x)
        x_tensor = torch.cuda.FloatTensor(x_set)
        y_tensor = torch.cuda.FloatTensor(y_set)
    else:
        x_line_tensor = torch.FloatTensor(x)
        x_tensor = torch.FloatTensor(x_set)
        y_tensor = torch.FloatTensor(y_set)

    # Training an ensemble of 5 networks(MLPS) with MSE
    # TODO:Draw Fig1.1
    if args.fig == 1:
        means = []
        for _ in range(5):
            model = MLP(x_tensor.size(1), num_hidden, num_layer)
            if is_cuda:
                model.cuda()

            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses = []
            for __ in tqdm(range(epochs), desc='fig_1 training...'):
                total_train_loss = 0

                optimizer.zero_grad()
                y_pred = model.forward(x_tensor)
                loss = loss_fn(y_pred, y_tensor)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                train_losses.append(total_train_loss)
            mean = model(x_line_tensor)
            means.append(mean.detach().numpy())

        # Have to calculte predicted mean and std
        # 'mean' have to be a numpy array with shape [100,1]
        # 'std'  have to be a numpy array with shape [100,1]
        means = np.array(means)
        mean = np.mean(means, axis=0)
        std = np.std(means, axis=0)

        draw_graph(x, x_set, y_set, mean, std)

    # Training a Gaussian MLP(single network) with NLL score rule
    # TODO:Draw Fig1.2
    elif args.fig == 2:
        means = []
        vars = []
        for _ in range(1):
            model = GaussianMLP(x_tensor.size(1), num_hidden, num_layer)
            if is_cuda:
                model.cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses = []
            for __ in tqdm(range(epochs), desc='fig_2 training...'):
                total_train_loss = 0

                optimizer.zero_grad()
                mu_pred, sigma_pred = model.forward(x_tensor)
                loss = model.NLLLoss(mu_pred, sigma_pred, y_tensor)
                # print(loss, x_tensor.data[0], mu_pred.data[0], sigma_pred.data[0])
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_losses.append(total_train_loss)
            mu_pred, sigma_pred = model(x_line_tensor)
            means.append(mu_pred.detach().numpy())
            vars.append(sigma_pred.detach().numpy())

        # Have to calculte predicted mean and var
        # 'mean' have to be a numpy array with shape [100,1]
        # 'var'  have to be a numpy array with shape [100,1]
        means = np.array(means)
        mean = np.mean(means, axis=0)
        var = np.mean(vars, axis=0)

        draw_graph(x, x_set, y_set, mean, np.sqrt(var))

    # Training a Gaussian MLP with NLL & Adversarial Training
    # TODO:Draw Fig1.3
    elif args.fig == 3:
        means = []
        vars = []
        for _ in range(1):
            model = GaussianMLP(x_tensor.size(1), num_hidden, num_layer)
            if is_cuda:
                model.cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses = []
            x_prime_tensor = torch.autograd.Variable(torch.FloatTensor(x_set))
            x_prime_tensor.requires_grad = True
            for __ in tqdm(range(epochs), desc='fig_2 training...'):
                total_train_loss = 0

                optimizer.zero_grad()
                mu_pred, sigma_pred = model.forward(x_prime_tensor)
                loss = model.NLLLoss(mu_pred, sigma_pred, y_tensor)
                loss.backward()
                optimizer.step()

                x_prime_tensor = x_tensor + (epsilon * x_prime_tensor.grad)
                x_prime_tensor.requires_grad = True

                total_train_loss += loss.item()
                train_losses.append(total_train_loss)
            mu_pred, sigma_pred = model(x_line_tensor)
            means.append(mu_pred.detach().numpy())
            vars.append(sigma_pred.detach().numpy())

        # Have to calculte predicted mean and var
        # 'mean' have to be a numpy array with shape [100,1]
        # 'var'  have to be a numpy array with shape [100,1]
        means = np.array(means)
        mean = np.nanmean(means, axis=0)
        var = np.nanmean(vars, axis=0)

        draw_graph(x, x_set, y_set, mean, np.sqrt(var))

    # Training a Gaussian mixture MLP (Deep ensemble) with NLL
    # TODO: Draw Fig1.4
    else:
        means = []
        vars = []
        for _ in range(20):
            model = GaussianMLP(x_tensor.size(1), num_hidden, num_layer)
            if is_cuda:
                model.cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses = []
            x_prime_tensor = torch.autograd.Variable(torch.FloatTensor(x_set))
            x_prime_tensor.requires_grad = True
            for j in tqdm(range(epochs), desc='fig_2 training...'):
                total_train_loss = 0

                optimizer.zero_grad()
                mu_pred, sigma_pred = model.forward(x_prime_tensor)
                loss = model.NLLLoss(mu_pred, sigma_pred, y_tensor)
                loss.backward()
                optimizer.step()

                x_prime_tensor = x_tensor + (epsilon * x_prime_tensor.grad)
                x_prime_tensor.requires_grad = True

                total_train_loss += loss.item()
                train_losses.append(total_train_loss)
            mu_pred, sigma_pred = model(x_line_tensor)
            means.append(mu_pred.detach().numpy())
            vars.append(sigma_pred.detach().numpy())

        # Have to calculte predicted mean and var
        # 'mean' have to be a numpy array with shape [100,1]
        # 'var'  have to be a numpy array with shape [100,1]
        means = np.array(means)
        vars = np.array(vars)
        mean = np.nanmean(means, axis=0)
        var = np.nanmean(np.power(vars, 2) + np.power(means, 2), axis=0) - np.power(mean, 2)

        draw_graph(x, x_set, y_set, mean, np.sqrt(var))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep ensemble')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10000)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20)

    parser.add_argument(
        '--fig',
        type=int,
        default=4)
    args = parser.parse_args()

    main(args)
