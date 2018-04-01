import os
import torch
import argparse
import numpy as np
import torch.utils.data

from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image


class AutoEncoder(nn.Module):
    def __init__(self, inp_size, hid_size):
        super(AutoEncoder, self).__init__()
        """
        Here you should define layers of your autoencoder
        Please note, if a layer has trainable parameters, it should be nn.Linear.
        ## !! CONVOLUTIONAL LAYERS CAN NOT BE HERE !! ##
        However, you can use any noise inducing layers, e.g. Dropout.

        Your network must not have more than six layers with trainable parameters.
        :param inp_size: integer, dimension of the input object
        :param hid_size: integer, dimension of the hidden representation
        """
        self.inp_size = inp_size
        self.hid_size = hid_size

        ################################################################
        # Hacky way to introduce hyper-parameters, since we can't modify
        # the functions or class inputs and have to fill only the blanks
        # I used some of these for research question purposes
        self.num_layers = 3 # Numer of layers
        self.l1_loss = True # or L2 alternative
        self.l1_weights = True # or L2 regularisation alternative
        self.lam = 0.0001 # Parameter regularisation strength
        self.loss_f = nn.L1Loss() if self.l1_loss == True else nn.MSELoss()
        self.weight_f = nn.L1Loss(size_average=False) if self.l1_weights == True else nn.MSELoss(size_average=False)

        # Let the encoder and decoder number of hidden units be adjusted
        # according to local hyper-parameter - number of layers
        encoder_l = np.linspace(self.inp_size, self.hid_size, self.num_layers+1).astype(int).tolist()
        decoder_l = encoder_l[::-1]

        # Build a list of tuples (in_size, out_size) for nn.Linear
        self.encoder_l = [(encoder_l[i], encoder_l[i+1]) for i in range(len(encoder_l[:-1]))]
        self.decoder_l = [(decoder_l[i], decoder_l[i+1]) for i in range(len(decoder_l[:-1]))]

        # Given above build encoder and decoder networks
        self.encoder = self.return_mlp(self.num_layers, self.encoder_l)
        self.decoder = self.return_mlp(self.num_layers, self.decoder_l)

    @staticmethod
    def return_mlp(num_layers, num_hidden):
        """
        Applicant defined function to return an mlp

        :param num_layers: int, number of layers
        :param num_hidden: list, with elements being a number of hidden units
        """
        # Creates layers in an order Linear, Tanh, Linear, Tanh,.. and so on.. using list comprehension
        layers = [[nn.Linear(num_hidden[i][0], num_hidden[i][1]), nn.BatchNorm1d(num_hidden[i][1]),
                   nn.ReLU()] for i in range(num_layers-1)]
        layers = [layer for sublist in layers for layer in sublist]

        # Append last layer whihc will be just Linear in this case
        layers.append(nn.Linear(num_hidden[num_layers-1][0], num_hidden[num_layers-1][1]))
        layers.append(nn.Sigmoid())

        # Convert into model
        model = nn.Sequential(*layers)

        return model

    def param_reg(self):
        """
        Applies regularisation to model parameters
        """
        reg = 0

        # Loop over models and their parameters and compute regularisation constraints
        for model in [self.encoder, self.decoder]:
            for param in model.parameters():
                ## NOTE: REMOVE CUDA BEFORE SUBMITTING
                target = Variable(torch.zeros(param.size()).cuda(0))
                reg += self.weight_f(param, target)

        # Multiply with regularisation strenght and return
        return reg * self.lam

    def encode(self, x):
        """
        Encodes objects to hidden representations (E: R^inp_size -> R^hid_size)

        :param x: inputs, Variable of shape (batch_size, inp_size)
        :return:  hidden represenation of the objects, Variable of shape (batch_size, hid_size)
        """
        return self.encoder(x)

    def decode(self, h):
        """
        Decodes objects from hidden representations (D: R^hid_size -> R^inp_size)

        :param h: hidden represenatations, Variable of shape (batch_size, hid_size)
        :return:  reconstructed objects, Variable of shape (batch_size, inp_size)
        """
        return self.decoder(h)

    def forward(self, x):
        """
        Encodes inputs to hidden representations and decodes back.

        x: inputs, Variable of shape (batch_size, inp_size)
        return: reconstructed objects, Variable of shape (batch_size, inp_size)
        """
        return self.decode(self.encode(x))

    def loss_function(self, recon_x, x):
        """
        Calculates the loss function.

        :params recon_x: reconstructed object, Variable of shape (batch_size, inp_size)
        :params x: original object, Variable of shape (batch_size, inp_size)
        :return: loss
        """
        loss = self.loss_f(recon_x, x)

        reg_loss = self.param_reg()

        return loss + reg_loss

def train(model, optimizer, train_loader, test_loader):
    for epoch in range(10):
        model.train()
        train_loss, test_loss = 0, 0
        for data, _ in train_loader:
            data = Variable(data).view(-1, 784)
            x_rec = model(data)
            loss = model.loss_function(x_rec, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        print('=> Epoch: %s Average loss: %.3f' % (epoch, train_loss / len(train_loader.dataset)))

        model.eval()
        for data, _ in test_loader:
            data = Variable(data, volatile=True).view(-1, 784)
            x_rec = model(data)
            test_loss += model.loss_function(x_rec, data).data[0]

        test_loss /= len(test_loader.dataset)
        print('=> Test set loss: %.3f' % test_loss)

        n = min(data.size(0), 8)
        comparison = torch.cat([data.view(-1, 1, 28, 28)[:n], x_rec.view(-1, 1, 28, 28)[:n]])
        if not os.path.exists('./pics'): os.makedirs('./pics')
        save_image(comparison.data.cpu(), 'pics/reconstruction_' + str(epoch) + '.png', nrow=n)
    return model


def test_work():
    print('Start test')
    get_loader = lambda train: torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=train, download=True, transform=transforms.ToTensor()),
        batch_size=50, shuffle=True)
    train_loader, test_loader = get_loader(True), get_loader(False)

    try:
        model = AutoEncoder(inp_size=784, hid_size=20)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    except Exception:
        assert False, 'Error during model creation'
        return

    try:
        model = train(model, optimizer, train_loader, test_loader)
    except Exception:
        assert False, 'Error during training'
        return

    test_x = Variable(torch.randn(1, 784))
    rec_x, hid_x = model(test_x), model.encode(test_x)
    submodules = dict(model.named_children())
    layers_with_params = np.unique(['.'.join(n.split('.')[:-1]) for n, _ in model.named_parameters()])

    assert (hid_x.dim() == 2) and (hid_x.size(1) == 20),  'Hidden representation size must be equal to 20'
    assert (rec_x.dim() == 2) and (rec_x.size(1) == 784), 'Reconstruction size must be equal to 784'
    assert len(layers_with_params) <= 6, 'The model must have no more than 6 layers '
    assert np.all(np.concatenate([list(p.shape) for p in model.parameters()]) <= 800), 'All hidden sizes must be less than 800'
    assert np.all([isinstance(submodules[name], nn.Linear) for name in layers_with_params]), 'All layers with parameters must be nn.Linear'
    print('Success!🎉')

if __name__ == '__main__':
    test_work()
