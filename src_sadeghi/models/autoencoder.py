import torch


class AutoEncoderBasic(torch.nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(AutoEncoderBasic, self).__init__()
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.output_size = input_size

        self.fc1 = torch.nn.Linear(self.input_size, self.encoding_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.encoding_dim, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def encoder(self, x):
        return self.relu(self.fc1(x))

    def decoder(self, y):
        return self.sigmoid(self.fc2(y))

    def forward(self, x):
        y = self.encoder(x.reshape(-1, self.input_size))
        x = self.decoder(y)
        return x


class AutoEncoderLinear(torch.nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(AutoEncoderLinear, self).__init__()
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.output_size = input_size

        self.fc1 = torch.nn.Linear(self.input_size, self.encoding_dim)
        self.fc2 = torch.nn.Linear(self.encoding_dim, self.output_size)

    def encoder(self, x):
        return self.fc1(x)

    def decoder(self, y):
        return self.fc2(y)

    def forward(self, x):
        y = self.encoder(x.reshape(-1, self.input_size))
        x = self.decoder(y)
        return x


def get_autoencoder(dtype):
    if dtype == 'linear':
        return AutoEncoderLinear
    else:
        return AutoEncoderBasic
