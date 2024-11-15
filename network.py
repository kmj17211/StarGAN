import torch
from torch.nn import *

def expand_cls(size, cls):
    cls = cls.view(cls.size(0), cls.size(1), 1, 1)
    return cls.repeat(1, 1, size[2], size[3])

class Residual_Block(Module):
    def __init__(self, channel):
        super().__init__()

        block_layer = []
        
        block_layer.append(Conv2d(channel, channel, kernel_size = 3, padding = 1, bias = False))
        block_layer.append(InstanceNorm2d(channel, affine = True))
        block_layer.append(ReLU(inplace = True))
        block_layer.append(Conv2d(channel, channel, kernel_size = 3, padding = 1, bias = False))
        block_layer.append(InstanceNorm2d(channel, affine = True))

        self.block_layer = Sequential(*block_layer)

    def forward(self, x):
        x = self.block_layer(x) + x
        return x

class Generator(Module):
    def __init__(self, in_channel = 2, out_channel = 1, repeat_bottle_neck = 6):
        super().__init__()
        
        conv_dim = 64

        down_layer = []
        # 128 x 128 x 2
        down_layer.append(Conv2d(in_channel, conv_dim, kernel_size = 7, padding = 3, bias = False))
        down_layer.append(InstanceNorm2d(conv_dim, affine = True))
        down_layer.append(ReLU(inplace = True))
        # 128 x 128 x 64
        down_layer.append(Conv2d(conv_dim, conv_dim*2, kernel_size = 4, stride = 2, padding = 1, bias = False))
        down_layer.append(InstanceNorm2d(conv_dim*2, affine = True))
        down_layer.append(ReLU(inplace = True))
        # 64 x 64 x 128
        down_layer.append(Conv2d(conv_dim*2, conv_dim*4, kernel_size = 4, stride = 2, padding = 1, bias = False))
        down_layer.append(InstanceNorm2d(conv_dim*4, affine = True))
        down_layer.append(ReLU(inplace = True))

        bottle_neck = []
        # 32 x 32 x 256
        for n in range(repeat_bottle_neck):
            bottle_neck.append(Residual_Block(conv_dim*4))
        
        up_layer = []
        # 32 x 32 x 256
        up_layer.append(ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size = 4, stride = 2, padding = 1, bias = False))
        up_layer.append(InstanceNorm2d(conv_dim*2, affine = True))
        up_layer.append(ReLU(inplace = True))
        # 64 x 64 x 128
        up_layer.append(ConvTranspose2d(conv_dim*2, conv_dim, kernel_size = 4, stride = 2, padding = 1, bias = False))
        up_layer.append(InstanceNorm2d(conv_dim, affine = True))
        up_layer.append(ReLU(inplace = True))
        # 128 x 128 x 64
        up_layer.append(Conv2d(conv_dim, out_channel, kernel_size = 7, padding = 3, bias = False))
        up_layer.append(Tanh())

        self.down_layer = Sequential(*down_layer)
        self.bottle_neck = Sequential(*bottle_neck)
        self.up_layer = Sequential(*up_layer)

    def forward(self, x, cls):
        cls = expand_cls(x.shape, cls)
        x = torch.cat((x, cls), dim = 1)
        x = self.down_layer(x)
        x = self.bottle_neck(x)
        x = self.up_layer(x)
        return x
    
class Discriminator(Module):
    def __init__(self, in_channel = 3, cls_dim = 6, WGAN = False):
        super().__init__()

        conv_dim = 64

        layer = []
        # 128 x 128 x 1
        layer.append(Conv2d(in_channel, conv_dim, kernel_size = 4, stride = 2, padding = 1))
        layer.append(LeakyReLU(0.01, inplace = True))
        # 64 x 64 x 64
        if WGAN:
            layer.append(Conv2d(conv_dim, conv_dim*2, kernel_size = 4, stride = 2, padding = 1))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 32 x 32 x 128
            layer.append(Conv2d(conv_dim*2, conv_dim*4, kernel_size = 4, stride = 2, padding = 1))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 16 x 16 x 256
            layer.append(Conv2d(conv_dim*4, conv_dim*8, kernel_size = 4, stride = 2, padding = 1))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 8 x 8 x 512
            layer.append(Conv2d(conv_dim*8, conv_dim*16, kernel_size = 4, stride = 2, padding = 1))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 4 x 4 x 1024
            layer.append(Conv2d(conv_dim*16, conv_dim*32, kernel_size = 4, stride = 2, padding = 1))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 2 x 2 x 2048
        else:
            layer.append(Conv2d(conv_dim, conv_dim*2, kernel_size = 4, stride = 2, padding = 1))
            layer.append(InstanceNorm2d(conv_dim*2))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 32 x 32 x 128
            layer.append(Conv2d(conv_dim*2, conv_dim*4, kernel_size = 4, stride = 2, padding = 1))
            layer.append(InstanceNorm2d(conv_dim*4))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 16 x 16 x 256
            layer.append(Conv2d(conv_dim*4, conv_dim*8, kernel_size = 4, stride = 2, padding = 1))
            layer.append(InstanceNorm2d(conv_dim*8))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 8 x 8 x 512
            layer.append(Conv2d(conv_dim*8, conv_dim*16, kernel_size = 4, stride = 2, padding = 1))
            layer.append(InstanceNorm2d(conv_dim*16))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 4 x 4 x 1024
            layer.append(Conv2d(conv_dim*16, conv_dim*32, kernel_size = 4, stride = 2, padding = 1))
            layer.append(InstanceNorm2d(conv_dim*32))
            layer.append(LeakyReLU(0.01, inplace = True))
            # 2 x 2 x 2048
        
        self.main_layer = Sequential(*layer)

        self.layer_dis = Conv2d(conv_dim*32, 1, kernel_size = 3, padding = 1, bias = False)
        # 2 x 2 x 1
        self.layer_cls = Conv2d(conv_dim*32, cls_dim, kernel_size = 2, bias = False)
        # 1 x 1 x cls_dim

    def forward(self, x):
        x = self.main_layer(x)
        out_dis = self.layer_dis(x)
        out_cls = self.layer_cls(x)
        # out_dis = torch.sigmoid(out_dis)
        # out_cls = torch.sigmoid(out_cls)
        return out_dis, out_cls.view(out_cls.size(0), out_cls.size(1))
