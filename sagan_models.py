import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

from quantization_modules import Conv2dLSQ, TransposeConv2dLSQ, ActLSQ

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation, nbits=3):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = Conv2dLSQ(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1, nbits=nbits)
        self.key_conv = Conv2dLSQ(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1, nbits=nbits)
        self.value_conv = Conv2dLSQ(in_channels = in_dim , out_channels = in_dim , kernel_size= 1, nbits=nbits)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, nbits=3, nbits_act=3):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(TransposeConv2dLSQ(z_dim, conv_dim * mult, 4, nbits=nbits)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(TransposeConv2dLSQ(curr_dim, int(curr_dim / 2), 4, 2, 1, nbits=nbits)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(TransposeConv2dLSQ(curr_dim, int(curr_dim / 2), 4, 2, 1, nbits=nbits)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(TransposeConv2dLSQ(curr_dim, int(curr_dim / 2), 4, 2, 1, nbits=nbits)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(TransposeConv2dLSQ(curr_dim, 3, 4, 2, 1, nbits=nbits))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

        self.actQ1 = ActLSQ(nbits=nbits_act)
        self.actQ2 = ActLSQ(nbits=nbits_act)
        self.actQ3 = ActLSQ(nbits=nbits_act)
        self.actQ4 = ActLSQ(nbits=nbits_act)
        self.actQ5 = ActLSQ(nbits=nbits_act)
        self.actQ6 = ActLSQ(nbits=nbits_act)
        self.actQ7 = ActLSQ(nbits=nbits_act)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        #act1
        out=self.actQ1(z)
        out=self.l1(out)
        #act2
        out=self.actQ2(out)
        out=self.l2(out)
        #act3
        out=self.actQ3(out)
        out=self.l3(out)
        #act4
        out=self.actQ4(out)
        out,p1 = self.attn1(out)
        #act5
        out=self.actQ5(out)
        out=self.l4(out)
        #act6
        out=self.actQ6(out)
        out,p2 = self.attn2(out)
        #act7
        out=self.actQ7(out)
        out=self.last(out)

        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64, nbits=3, nbits_act=3):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(Conv2dLSQ(3, conv_dim, 4, 2, 1, nbits=nbits)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(Conv2dLSQ(curr_dim, curr_dim * 2, 4, 2, 1, nbits=nbits)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(Conv2dLSQ(curr_dim, curr_dim * 2, 4, 2, 1, nbits=nbits)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(Conv2dLSQ(curr_dim, curr_dim * 2, 4, 2, 1, nbits=nbits)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(Conv2dLSQ(curr_dim, 1, 4, nbits=nbits))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

        self.actQ1 = ActLSQ(nbits=nbits_act)
        self.actQ2 = ActLSQ(nbits=nbits_act)
        self.actQ3 = ActLSQ(nbits=nbits_act)
        self.actQ4 = ActLSQ(nbits=nbits_act)
        self.actQ5 = ActLSQ(nbits=nbits_act)
        self.actQ6 = ActLSQ(nbits=nbits_act)
        self.actQ7 = ActLSQ(nbits=nbits_act)

    def forward(self, x):
        #act1
        out=self.actQ1(x)
        out = self.l1(out)
        #act2
        out=self.actQ2(out)
        out = self.l2(out)
        #act3
        out=self.actQ3(out)
        out = self.l3(out)
        #act4
        out=self.actQ4(out)
        out,p1 = self.attn1(out)
        #act5
        out=self.actQ5(out)
        out=self.l4(out)
        #act6
        out=self.actQ6(out)
        out,p2 = self.attn2(out)
        #act7
        out=self.actQ7(out)
        out=self.last(out)

        return out.squeeze(), p1, p2
