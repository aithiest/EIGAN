import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorCNN(nn.Module):
    def __init__(self, ngpu, nc, ndf, output_dim):
        super(GeneratorCNN, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 2, 4, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 2, 4, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 2, 4, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, output_dim, 2, 4, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).squeeze(2).squeeze(2)


class GeneratorFCN(torch.nn.Module):
    def __init__(
        self, input_size, output_size,
    ):
        super(GeneratorFCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, 32)
        self.fc2 = torch.nn.Linear(32, self.output_size)

    def forward(self, x):
        output = self.fc1(x.reshape(-1, self.input_size))
        output = F.leaky_relu(output)
        output = self.fc2(output)
        output = torch.sigmoid(output)

        return output


class DiscriminatorFCN(torch.nn.Module):
    def __init__(
        self, input_size, output_size,
    ):
        super(DiscriminatorFCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, 512)
        self.fc2 = torch.nn.Linear(512, self.output_size)

    def forward(self, x):
        output = self.fc1(x.reshape(-1, self.input_size))
        output = F.relu(output)
        output = self.fc2(output)
        output = torch.sigmoid(output)

        return output


class GeneratorLogistic(torch.nn.Module):
    def __init__(
        self, input_size, output_size,
    ):
        super(GeneratorLogistic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        output = self.fc1(x.reshape(-1, self.input_size))
        output = torch.sigmoid(output)

        return output


class DiscriminatorLogistic(torch.nn.Module):
    def __init__(
        self, input_size, output_size,
    ):
        super(DiscriminatorLogistic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        output = self.fc1(x.reshape(-1, self.input_size))
        output = torch.sigmoid(output)

        return output


class GeneratorLinear(torch.nn.Module):
    def __init__(
        self, input_size, output_size,
    ):
        super(GeneratorLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        return self.fc1(x.reshape(-1, self.input_size))


class DiscriminatorLinear(torch.nn.Module):
    def __init__(
        self, input_size, output_size,
    ):
        super(DiscriminatorLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        output = self.fc1(x.reshape(-1, self.input_size))

        return output


def get_disrimininator(dtype):
    if dtype.lower() == 'fcn':
        return DiscriminatorFCN
    elif dtype.lower() == 'logistic':
        return DiscriminatorLogistic
    elif dtype.lower() == 'linear':
        return DiscriminatorLinear
