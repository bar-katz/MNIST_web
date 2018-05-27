import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib.legend_handler import HandlerLine2D
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import matplotlib.pyplot as plt
import argparse

__author__ = 'Bar Katz'


class NeuralNetBasic(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetBasic, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, hidden1_size)
        self.fc1 = nn.Linear(hidden1_size, hidden2_size)
        self.fc2 = nn.Linear(hidden2_size, mnist_output_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return f.log_softmax(x, dim=1)


class NeuralNetDropout(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetDropout, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, hidden1_size)
        self.fc1 = nn.Linear(hidden1_size, hidden2_size)
        self.fc2 = nn.Linear(hidden2_size, mnist_output_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0(x))
        x = f.relu(self.fc1(x))
        x = f.dropout(x, 0.2, self.training)
        x = f.relu(self.fc2(x))
        x = f.dropout(x, 0.2, self.training)
        return f.log_softmax(x, dim=1)


class NeuralNetBatchNorm(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetBatchNorm, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, hidden1_size)
        self.fc0_bn = nn.BatchNorm1d(hidden1_size)
        self.fc1 = nn.Linear(hidden1_size, hidden2_size)
        self.fc1_bn = nn.BatchNorm1d(hidden2_size)
        self.fc2 = nn.Linear(hidden2_size, mnist_output_size)
        self.fc2_bn = nn.BatchNorm1d(mnist_output_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0_bn(self.fc0(x)))
        x = f.relu(self.fc1_bn(self.fc1(x)))
        x = f.relu(self.fc2_bn(self.fc2(x)))
        return f.log_softmax(x, dim=1)


class NeuralNetConv(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetConv, self).__init__()
        self.image_size = image_size

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc0 = nn.Linear(320, hidden1_size)
        self.fc1 = nn.Linear(hidden1_size, hidden2_size)
        self.fc2 = nn.Linear(hidden2_size, mnist_output_size)

    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 320)
        x = f.relu(self.fc0(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return f.log_softmax(x, dim=1)


class NeuralNetCombine(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetCombine, self).__init__()
        self.image_size = image_size

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc0 = nn.Linear(320, hidden1_size)
        self.fc0_bn = nn.BatchNorm1d(hidden1_size)
        self.fc1 = nn.Linear(hidden1_size, hidden2_size)
        self.fc1_bn = nn.BatchNorm1d(hidden2_size)
        self.fc2 = nn.Linear(hidden2_size, mnist_output_size)
        self.fc2_bn = nn.BatchNorm1d(mnist_output_size)

    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = f.relu(self.fc0_bn(self.fc0(x)))
        x = f.relu(self.fc1_bn(self.fc1(x)))
        x = f.dropout(x, 0.2, self.training)
        x = f.relu(self.fc2_bn(self.fc2(x)))
        x = f.dropout(x, 0.2, self.training)
        return f.log_softmax(x, dim=1)


def train(epoch, model, train_loader, optimizer):
    model.train()

    train_loss = 0
    correct_train = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct_train += pred.eq(labels.data.view_as(pred)).cpu().sum()

    train_loss /= len(train_loader)
    print('Train Epoch: {}\nAccuracy {}/{} ({:.0f}%)\nAverage loss: {:.6f}'.format(
        epoch, correct_train, len(train_loader) * batch_size,
        100. * correct_train / (len(train_loader) * batch_size), train_loss))

    return train_loss


def validation(epoch, model, valid_loader):
    model.eval()

    valid_loss = 0
    correct_valid = 0
    for data, label in valid_loader:
        output = model(data)
        valid_loss += f.nll_loss(output, label, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct_valid += pred.eq(label.data.view_as(pred)).cpu().sum()

    valid_loss /= (len(valid_loader) * batch_size)
    print('Validation Epoch: {}\nAccuracy: {}/{} ({:.0f}%)\nAverage loss: {:.6f}'.format(
        epoch, correct_valid, (len(valid_loader) * batch_size),
        100. * correct_valid / (len(valid_loader) * batch_size), valid_loss))

    return valid_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    predictions = list()
    for data, target in test_loader:
        output = model(data)
        test_loss += f.nll_loss(output, target, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        pred_vec = pred.view(len(pred))
        for x in pred_vec:
            predictions.append(x.item())
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set:\nAccuracy: {}/{} ({:.0f}%)\nAverage loss: {:.4f}'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), test_loss))

    return predictions


nn_dict = {'Basic': NeuralNetBasic,
           'Dropout': NeuralNetDropout,
           'Batch_norm': NeuralNetBatchNorm,
           'Conv': NeuralNetConv,
           'Combine': NeuralNetCombine}

# consts
mnist_output_size = 10
mnist_image_size = 28 * 28


# parameters
epochs = 10
learning_rate = 0.01
batch_size = 64
valid_split = 0.2

neural_net = nn_dict['Basic']
hidden1_size = 100
hidden2_size = 50

write_test_pred = False
draw_loss_graph = False


def init_params():
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Neural nets with 2 hidden layers using pytorch on fashionMNIST data set')

    # Add arguments
    parser.add_argument(
        '-a', '--neural_net', type=str, help='Neural net', required=False, default='Basic')
    parser.add_argument(
        '-e', '--epochs', type=int, help='Number of epochs', required=False, default=10)
    parser.add_argument(
        '-l', '--learning_rate', type=float, help='Learning rate', required=False, default=0.01)
    parser.add_argument(
        '-b', '--batch_size', type=int, help='Batch size', required=False, default=64)
    parser.add_argument(
        '-s', '--validation_split', type=float, help='Percent of data to be used as validation', required=False,
        default=0.2)
    parser.add_argument(
        '-h1', '--hidden1_size', type=int, help='First hidden layer size', required=False, default=100)
    parser.add_argument(
        '-h2', '--hidden2_size', type=int, help='Second hidden layer size', required=False, default=50)
    parser.add_argument(
        '-w', '--write', type=bool, help='Write test set predictions to file', required=False, default=False)
    parser.add_argument(
        '-d', '--draw', type=bool, help='Draw validation and train loss graph', required=False, default=False)

    # Array for all arguments passed to script
    args = parser.parse_args()

    # Assign parameters
    global neural_net
    global epochs
    global learning_rate
    global batch_size
    global valid_split
    global hidden1_size
    global hidden2_size
    global write_test_pred
    global draw_loss_graph

    neural_net = nn_dict[args.neural_net]
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    valid_split = args.validation_split
    hidden1_size = args.hidden1_size
    hidden2_size = args.hidden2_size
    write_test_pred = args.write
    draw_loss_graph = args.draw


def get_data_loaders():
    tran = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_ds = datasets.FashionMNIST('./train_data', train=True, download=True, transform=tran)
    test_ds = datasets.FashionMNIST('./test_data', train=False, download=True, transform=tran)

    num_train = len(train_ds)
    indices = list(range(num_train))
    split = int(np.floor(valid_split * num_train))

    valid_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def train_model(model, train_loader, valid_loader, test_loader):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    x = list()
    train_y = list()
    valid_y = list()
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer)
        valid_loss = validation(epoch, model, valid_loader)

        x.append(epoch)
        train_y.append(train_loss)
        valid_y.append(valid_loss)
    predictions = test(model, test_loader)

    options(x, train_y, valid_y, predictions)


def options(x, train_y, valid_y, predictions):
    if write_test_pred:
        write_to_file(predictions)

    if draw_loss_graph:
        draw_loss(x, train_y, valid_y)


def write_to_file(predictions):
    # write prediction on test set to a file
    prev_x = None

    file = open('test.pred', 'w')
    for x in predictions:
        if prev_x is not None:
            file.write(str(prev_x) + '\n')
        prev_x = x

    file.write(str(prev_x))
    file.close()


def draw_loss(x, train_y, valid_y):
    fig = plt.figure(0)
    fig.canvas.set_window_title('Train loss VS Validation loss')
    plt.axis([0, 11, 0, 2])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    train_graph, = plt.plot(x, train_y, 'r--', label='Train loss')

    plt.plot(x, valid_y, label='Validation loss')

    plt.legend(handler_map={train_graph: HandlerLine2D(numpoints=4)})
    plt.show()


def main():

    init_params()

    train_loader, valid_loader, test_loader = get_data_loaders()

    model = neural_net(image_size=mnist_image_size)

    train_model(model, train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    main()
