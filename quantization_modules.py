import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import math

def grad_scale(x, scale):
    yOut = x
    yGrad = x*scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def round_pass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def quantizeLSQ(v, s, p, isActivation=False):
    if isActivation:
        Qn = 0
        Qp = 2**p - 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)
    else: # is weight
        Qn = -2**(p-1)
        Qp = 2**(p-1) - 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)

    #quantize
    s = grad_scale(s, gradScaleFactor)
    vbar = round_pass((v/s).clamp(Qn, Qp))
    vhat = vbar*s
    return vhat

class TransposeConv2dLSQ(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, **kwargs_q):
        super(TransposeConv2dLSQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)

        self.nbits = kwargs_q['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        w_q = quantizeLSQ(self.weight, self.step_size, self.nbits)

        out = nn.functional.conv_transpose2d(x, w_q, None, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class Conv2dLSQ(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(Conv2dLSQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.nbits = kwargs['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        w_q = quantizeLSQ(self.weight, self.step_size, self.nbits)

        out = nn.functional.conv2d(input, w_q, None, self.stride, self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class ActLSQ(nn.Module):
    def __init__(self, **kwargs):
        super(ActLSQ, self).__init__()

        self.nbits = kwargs['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        x_q = quantizeLSQ(x, self.step_size, self.nbits, isActivation=True)

        return x_q
