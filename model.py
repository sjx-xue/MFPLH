import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class Extractor(nn.Module):
    def __init__(self, args, x, G):
        super(Extractor,self).__init__()

        self.args = args

        self.conv1 = torch.nn.Conv1d(in_channels=self.args.embedding_size,
                                     out_channels=64,
                                     kernel_size=2,
                                     stride=1
                                     )
        self.conv2 = torch.nn.Conv1d(in_channels=self.args.embedding_size,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.conv3 = torch.nn.Conv1d(in_channels=self.args.embedding_size,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=1
                                     )
        self.conv4 = torch.nn.Conv1d(in_channels=self.args.embedding_size,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=1
                                     )
        self.dropout = torch.nn.Dropout(0.6)
        self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=5)

        self.label_extractor = HGNN(in_ch=768, n_hid=2304)
        self.x = x
        self.G = G


    def forward(self, **kwargs):

        seqvec = kwargs['seqvec']
        evolution = kwargs['evolution']
        onehot = kwargs['onehot']
        length = kwargs['length']
        struct = kwargs['struct']

        x = torch.cat([seqvec, evolution], dim=-1)
        x = x.permute(0, 2, 1)

        x1 = self.conv1(x)
        x1 = torch.nn.ReLU()(x1)
        x1 = self.MaxPool1d(x1)

        x2 = self.conv2(x)
        x2 = torch.nn.ReLU()(x2)
        x2 = self.MaxPool1d(x2)

        x3 = self.conv3(x)
        x3 = torch.nn.ReLU()(x3)
        x3 = self.MaxPool1d(x3)

        x4 = self.conv4(x)
        x4 = torch.nn.ReLU()(x4)
        x4 = self.MaxPool1d(x4)

        y = torch.cat([x1, x2, x3, x4], dim=-1)
        representation = self.dropout(y)
        representation = representation.view(representation.size(0), -1)

        label_matrix = self.label_extractor(x=self.x, G=self.G)
        label_matrix_mean = torch.mean(label_matrix, dim=0)
        label_matrix_max, _ = torch.max(label_matrix, dim=0)

        result = representation + label_matrix_mean + label_matrix_max

        return result


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid):
        super(HGNN, self).__init__()

        self.dropout = torch.nn.Dropout(0.3)
        self.hgc1 = HGNN_conv(in_ch, 1024)
        self.hgc2 = HGNN_conv(1024,  n_hid)
        self.relu = torch.nn.ReLU()

    def forward(self, **kwargs):

        x = kwargs['x']
        G = kwargs['G']


        x = self.hgc1(x, G)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.hgc2(x, G)
        x = self.relu(x)
        return x


class Classifier_TDE_GELU(nn.Module):

    def __init__(self, args):
        super(Classifier_TDE_GELU, self).__init__()

        self.args = args
        self.use_effect = True
        self.full1 = nn.Linear(self.args.linear_size, 1000)
        self.full2 = nn.Linear(1000, 500)
        self.full3 = nn.Linear(500, 256)
        self.full4 = nn.Linear(256, 64)
        self.full5 = nn.Linear(64, self.args.label_size)

    def forward(self, representation, embed_mean):
        representation = representation.squeeze()

        if (not self.training) and self.use_effect:
            embed_mean = torch.from_numpy(embed_mean).view(1, -1).to(representation.device)
            representation = representation - 2 * (embed_mean / self.args.length)

        logit = self.full1(representation)
        logit = F.gelu(logit)

        logit = self.full2(logit)
        logit = F.gelu(logit)

        logit = self.full3(logit)
        logit = F.gelu(logit)

        logit = self.full4(logit)
        logit = F.gelu(logit)

        logit = self.full5(logit)

        return logit
