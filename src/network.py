# ==============================================================================
# sinc(i) - http://sinc.unl.edu.ar/
# L. Bugnon, C. Yones, J. Raad, M. Gerard, M. Rubiolo, G. Merino, M. Pividori,
# L. Di Persia, D.H. Milone, and G. Stegmayer.
# lbugnon@sinc.unl.edu.ar
# ==============================================================================
import torch.nn as nn

input_channels = 306
input_length = 2**14


class network(nn.Module):
    def __init__(self, hyperparams=None):
        """
        Constructor that defines the architecture of the deep neural network.
        """

        super(network, self).__init__()

        if hyperparams is None:
            nfilters = 12
            nlayers = 6
            ksize = 3
            self.use_Iblock = True
            self.fc_layer = False
        else:
            nfilters = hyperparams["nfilters"]
            nlayers = hyperparams["nlayers"]
            ksize = hyperparams["kernel_size"]
            self.use_Iblock = hyperparams["use_Iblock"]
            self.fc_layer = hyperparams["fc_layer"]

        self.outlen = input_length // 2**nlayers
        layers = []
        layers.append(nn.Conv1d(input_channels, int(2*nfilters), kernel_size=ksize, padding=1))
        layers.append(nn.ELU())
        layers.append(nn.BatchNorm1d(int(2*nfilters)))
        layers.append(nn.Conv1d(int(2*nfilters), nfilters, kernel_size=ksize, padding=1))
        for i in range(nlayers):
            layers.append(ResNet([nfilters, nfilters], [ksize, ksize]))
            layers.append(nn.AvgPool1d(2))
        layers.append(nn.ELU())
        layers.append(nn.BatchNorm1d(nfilters))
        layers.append(nn.Conv1d(nfilters, 1, kernel_size=1, padding=0))
        if self.fc_layer:
            layers.append(nn.Linear(self.outlen, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Feedforward function
        """
        x = self.layers(x)
        if not self.fc_layer:
            x = x.max(dim=2)[0]
        return x

    def heatmap(self, x):
        """
        Activation map given input x
        """
        out = self.layers(x)
        return out.view(-1, self.outlen)


class ResNet(nn.Module):
    def __init__(self, nfilters, ksizes, use_Iblock=True):
        super(ResNet, self).__init__()
        self.in_dim = nfilters[len(nfilters)-1]
        self.use_Iblock = use_Iblock
        nfilters.insert(0, self.in_dim)
        layers=[]
        for i in range(len(nfilters)-1):
            layers.append(nn.ELU())
            layers.append(nn.BatchNorm1d(nfilters[i]))
            layers.append(nn.Conv1d(nfilters[i], nfilters[i+1],
                kernel_size=ksizes[i], padding=int(ksizes[i]/2)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_Iblock:
            return self.layers(x) + x
        else:
            return self.layers(x)
