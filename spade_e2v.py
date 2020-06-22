import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                if 'Float' in input_.type():
                    self.zero_tensors[state_size] = (
                        torch.zeros(state_size).to(input_.device).float(),
                        torch.zeros(state_size).to(input_.device).float()
                    )
                else:
                    self.zero_tensors[state_size] = (
                        torch.zeros(state_size).to(input_.device).half(),
                        torch.zeros(state_size).to(input_.device).half()
                    )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2):
        super(RecurrentConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.recurrent_block = ConvLSTM(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):

        x = self.relu(self.bn(self.conv0(x)))
        state = self.recurrent_block(x, prev_state)
        x = state[0]
        return x, state


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)
        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()

        # instance normalization
        # self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = nhidden
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.relu = nn.ReLU()

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[-2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class UpConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, nom=3, scale=2):
        super(UpConvLayer, self).__init__()

        self.in_plane = in_channels
        self.out_plane = out_channels
        self.scale = scale
        self.planes = self.out_plane * scale ** 2

        self.conv0 = nn.Conv2d(self.in_plane, self.planes, kernel_size=5, padding=2, bias=True)
        self.icnr(scale=scale)
        self.shuf = nn.PixelShuffle(self.scale)

        self.norm = SPADE(self.out_plane, nom, nhidden)
        self.activation = nn.ReLU()

    def icnr(self, scale=2, init=nn.init.kaiming_normal_):
        ni, nf, h, w = self.conv0.weight.shape
        ni2 = int(ni / (scale ** 2))
        k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale ** 2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        self.conv0.weight.data.copy_(k)

    def forward(self, x, x_org):

        x = self.shuf(self.conv0(x))
        x = self.activation(self.norm(x, x_org))
        return x


class UpConvLayer3(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, nom=3):
        super(UpConvLayer3, self).__init__()

        self.in_plane = in_channels
        self.out_plane = out_channels
        self.scale = scale
        self.planes = self.out_plane * scale ** 2

        self.conv0 = nn.Conv2d(self.in_plane, self.planes, kernel_size=3, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(self.out_plane, self.out_plane, kernel_size=3, padding=1, bias=False)
        self.icnr(scale=scale)
        self.shuf = nn.PixelShuffle(self.scale)

        self.norm = SPADE(self.out_plane, nom)
        self.activation = nn.ReLU()

    def icnr(self, scale=2, init=nn.init.kaiming_normal_):
        ni, nf, h, w = self.conv0.weight.shape
        ni2 = int(ni / (scale ** 2))
        k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale ** 2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        self.conv0.weight.data.copy_(k)

    def forward(self, x, x_org):

        x = self.shuf(self.conv0(x))
        x = self.norm(x, x_org)
        x = self.activation(x)

        return x


class Unet6(nn.Module):

    def __init__(self):
        super(Unet6, self).__init__()

        self.fc = nn.Conv2d(5, 32, 5, padding=2)
        # layer 1
        self.rec0 = RecurrentConvLayer(32, 64, stride=1)
        self.rec1 = RecurrentConvLayer(64, 128, stride=2)
        self.rec2 = RecurrentConvLayer(128, 256, stride=2)

        # layer 2
        self.res0 = ResidualBlock(256, 256)
        self.res1 = ResidualBlock(256, 256)
        # layer 3
        self.up0 = UpConvLayer3(256, 128, nom=3)
        self.up1 = UpConvLayer3(128, 64, nom=3)
        self.up2 = RecurrentConvLayer(64, 32, stride=1)

        self.conv_img = nn.Conv2d(32, 3, kernel_size=1, padding=0)
        self.bn_img = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prev_states, pred):

        if prev_states is None:
            prev_states = [None] * 4

        x_org = pred
        head = self.relu(self.fc(x))
        # ------------
        x0, state0 = self.rec0(head, prev_states[0])
        x1, state1 = self.rec1(x0, prev_states[1])
        x2, state2 = self.rec2(x1, prev_states[2])
        # ------------
        x = self.res0(x2)
        x = self.res1(x)
        # ------------
        x = self.up0(x + x2, x_org)
        x = self.up1(x + x1, x_org)
        x, state3 = self.up2(x + x0, prev_states[3])
        # ------------

        stats = [state0, state1, state2, state3]
        # prediction layer and activation
        x = self.conv_img(self.relu(x + head))
        x = self.sigmoid(self.bn_img(x))
        return x, stats



