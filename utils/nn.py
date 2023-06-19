import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np



# ---
# Constants

LATENT_DIM = 512
MAPPING_NET_DEPTH = 8
MAX_CHANNELS = 512
MIN_WORKING_RESOLUTION = 8   # Used during progressive growing



# ---
# Networks

class Generator(nn.Module):

    def __init__(self, final_resolution=1024, prog_growth=False, device='cpu'):
        super().__init__()
        self.prog_growth = prog_growth
        self.mapping_net = MappingNetwork(device)
        self.synthesis_net = SynthesisNetwork(final_resolution, prog_growth, device)
        
    def forward(self, z):
        z = z / (z.square().mean() + 1e-8).sqrt()  # Normalize
        w = self.mapping_net(z)
        x = self.synthesis_net(w)
        return x


class MappingNetwork(nn.Module):

    def __init__(self, device):

        super().__init__()

        layers = []
        for _ in range(MAPPING_NET_DEPTH):
            layers.extend([
                Linear(LATENT_DIM, LATENT_DIM, bias_init=0.0, lr_multiplier=0.01),
                nn.LeakyReLU(0.2)
            ])
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, z):
        return self.model(z)


class SynthesisNetwork(nn.Module):

    def __init__(self, final_resolution, prog_growth, device):

        super().__init__()

        self.device = device
        
        self.prog_growth = prog_growth
        if prog_growth:
            working_resolution = MIN_WORKING_RESOLUTION
            self.has_unfused_new_block = False
            self.new_block = None
            self.to_rgb_skip = None
            self.alpha = None            
        else:
            working_resolution = final_resolution
        
        resolutions = [2**i for i in range(2, 12) if 2**i <= working_resolution]
        
        # Network body
        self.body = [SynthesisNetworkBlock(MAX_CHANNELS, MAX_CHANNELS, is_first_block=True, device=device)]
        in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        for res in resolutions[1:]:
            if res > 32: in_channels, out_channels = out_channels, out_channels // 2
            self.body.append(SynthesisNetworkBlock(in_channels, out_channels, device=device))
        
        # toRGB layer at the output resolution
        self.to_rgb = Conv2d(out_channels, 3, kernel_size=1, device=device)
        
        # Initial lowest res (4x4) feature map
        self.x_init = Parameter(torch.ones((1, MAX_CHANNELS, 4, 4), device=device))

    def forward(self, w):
        x = torch.repeat_interleave(self.x_init, repeats=w.shape[0], dim=0)
        for block in self.body:
            x = block(x, w)
        x = self._compute_output_image(x, w)
        return x       

    def _compute_output_image(self, x, w):
        if self.prog_growth:
            if self.has_unfused_new_block:
                x_main = self.new_block(x, w)
                x_main = self.to_rgb(x_main)
                x_skip = F.interpolate(x, scale_factor=2, mode='nearest')
                x_skip = self.to_rgb_skip(x_skip)
                x = (1 - self.alpha) * x_skip + self.alpha * x_main
            else:
                x = self.to_rgb(x)
        else:
            x = self.to_rgb(x)
        return x
    
    def set_alpha(self, alpha):
        self.alpha = alpha

    def grow_new_block(self):

        # Grow a new block at 2x res
        in_channels, out_channels = self.body[-1].out_channels, self.body[-1].out_channels // 2
        self.new_block = SynthesisNetworkBlock(in_channels, out_channels, device=self.device)

        # Replace current toRGB layer with 2x res one
        self.to_rgb = Conv2d(out_channels, 3, kernel_size=1, device=self.device)
        
        # Add a 2x res toRGB skip layer
        self.to_rgb_skip = Conv2d(in_channels, 3, kernel_size=1, device=self.device)
        
        # Reset alpha
        self.alpha = 0

        # Set flag
        self.has_unfused_new_block = True
    
    def fuse_new_block(self):
        
        # Fuse into the body
        self.body.append(self.new_block)
        
        # Cleanup
        self.new_block = None
        self.to_rgb_skip = None
        self.alpha = None
        
        # Clear flag
        self.has_unfused_new_block = False


class SynthesisNetworkBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_first_block=False, device='cpu'):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layers = []

        # 1st conv block
        if not is_first_block:
            self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.layers.append(Conv2d(in_channels, out_channels, kernel_size=3, padding=1, device=device))
        self.layers.extend([
            NoiseLayer(out_channels, device),
            AdaINLayer(out_channels, device),
            nn.LeakyReLU(0.2)
            ])
        
        # 2nd conv block
        self.layers.extend([
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, device=device),
            NoiseLayer(out_channels, device),
            AdaINLayer(out_channels, device),
            nn.LeakyReLU(0.2)
            ])
   
    def forward(self, x, w):
        for layer in self.layers:
            if isinstance(layer, AdaINLayer): x = layer(x, w)
            else:                             x = layer(x)
        return x
    

class NoiseLayer(nn.Module):
    
    def __init__(self, num_channels, device):
        super().__init__()
        self.scaling_factors = Parameter(torch.zeros((1, num_channels, 1, 1), device=device))

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        noise = torch.randn((batch_size, 1, height, width), device=x.device)
        return x + self.scaling_factors * noise


class AdaINLayer(nn.Module):
    
    def __init__(self, num_channels, device):
        super().__init__()
        self.affine_scale = Linear(LATENT_DIM, num_channels, bias_init=1.0, device=device)
        self.affine_bias = Linear(LATENT_DIM, num_channels, bias_init=0.0, device=device)
        self.eps = 1e-5

    def forward(self, x, w):
        style_scale = self.affine_scale(w).unsqueeze(-1).unsqueeze(-1)
        style_bias = self.affine_bias(w).unsqueeze(-1).unsqueeze(-1)
        mean = x.mean(dim=[-1, -2], keepdims=True)
        var = x.var(dim=[-1, -2], keepdims=True, unbiased=False) + self.eps
        x = style_scale * ((x - mean) / var.sqrt()) + style_bias
        return x


class Discriminator(nn.Module):

    def __init__(self, final_resolution=1024, prog_growth=False, device='cpu'):

        super().__init__()

        self.device = device

        self.prog_growth = prog_growth
        if prog_growth:
            working_resolution = MIN_WORKING_RESOLUTION
            self.has_unfused_new_block = False
            self.new_block = None
            self.to_rgb_skip = None
            self.alpha = None   
        else:
            working_resolution = final_resolution

        resolutions = [2**i for i in range(2, 12) if 2**i <= working_resolution]
        
        # Network body
        self.body = [DiscriminatorBlock(MAX_CHANNELS, MAX_CHANNELS, is_last_block=True, device=device)]  # Last block, corresponding to lowest res (4x4)
        in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        for res in resolutions[1:]: # Blocks laid out in reverse order - i.e. lo-res to hi-res
            if res > 32: in_channels, out_channels = in_channels // 2, in_channels
            self.body.append(DiscriminatorBlock(in_channels, out_channels, device=device))
        self.body = self.body[::-1]

        # fromRGB layer at the output resolution
        self.from_rgb = Conv2d(3, in_channels, kernel_size=1, device=device)

        # Last layer
        self.classifier = Linear(MAX_CHANNELS, 1, bias_init=0.0, device=device)

    def forward(self, x):
        x = self._compute_input_features(x)
        for block in self.body:
            x = block(x)
        pred = self.classifier(torch.reshape(x, (x.shape[0], -1)))
        return pred
    
    def _compute_input_features(self, x):
        if self.prog_growth:
            if self.has_unfused_new_block:
                x_main = self.from_rgb(x)
                x_main = self.new_block(x_main)
                x_skip = F.avg_pool2d(x, kernel_size=2, stride=2)
                x_skip = self.from_rgb_skip(x_skip)
                x = (1 - self.alpha) * x_skip + self.alpha * x_main
            else:
                x = self.from_rgb(x)
        else:
            x = self.from_rgb(x)
        return x
    
    def set_alpha(self, alpha):
        self.alpha = alpha

    def grow_new_block(self):

        # Grow a new block at 2x res
        in_channels, out_channels = self.body[0].in_channels // 2, self.body[0].in_channels
        self.new_block = DiscriminatorBlock(in_channels, out_channels, device=self.device)

        # Replace current fromRGB layer with 2x res one
        self.from_rgb = Conv2d(3, in_channels, kernel_size=1, device=self.device)
        
        # Add a 2x res fromRGB skip layer
        self.from_rgb_skip = Conv2d(3, out_channels, kernel_size=1, device=self.device)

        # Reset alpha
        self.alpha = 0
        
        # Set flag
        self.has_unfused_new_block = True
    
    def fuse_new_block(self):
        
        # Fuse into the body
        self.body.insert(0, self.new_block)
        
        # Cleanup
        self.new_block = None
        self.to_rgb_skip = None
        self.alpha = None
        
        # Clear flag
        self.has_unfused_new_block = False
    

class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_last_block=False, device='cpu'):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if is_last_block:
            layers = [
                MinibatchSDLayer(),
                Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                Conv2d(in_channels, out_channels, kernel_size=4),
                nn.LeakyReLU(0.2)
            ]
        else:
            layers= [
                Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=2, stride=2)
                ]

        self.block = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.block(x)



# ---
# Custom layers

class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias_init, lr_multiplier=1.0, device='cpu'):
        super().__init__()
        self.weight = Parameter(torch.randn([out_features, in_features], device=device) / lr_multiplier)
        self.bias = Parameter(torch.full([out_features], fill_value=bias_init, device=device))
        self.weight_gain = lr_multiplier * (1 / np.sqrt(in_features))  # For lr equalization + lr reduction
        self.bias_gain = lr_multiplier                                 #
    
    def forward(self, x):
        weight = self.weight * self.weight_gain
        bias = self.bias * self.bias_gain
        return F.linear(x, weight, bias)


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, padding_mode='reflect', device='cpu'):
        super().__init__()        
        self.weight = Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size], device=device))
        self.bias = Parameter(torch.full([out_channels], fill_value=0.0, device=device))
        self.weight_gain = 1 / np.sqrt(in_channels * kernel_size ** 2)   # For lr equalization
        self.padding = padding
        self.padding_mode = padding_mode

    def forward(self, x):
        weight = self.weight * self.weight_gain
        bias = self.bias
        x = F.pad(x, pad=[self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode)        
        return F.conv2d(x, weight, bias)        


class MinibatchSDLayer(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        sd = torch.mean(torch.sqrt(torch.var(x, dim=0, unbiased=False) + 1e-8))
        sd_map = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), fill_value=float(sd), device=x.device)
        x = torch.cat([x, sd_map], dim=1)
        return x



# ---
# Unit test
if __name__ == '__main__':
    
    device = 'cuda'
    final_resolution = 1024
    prog_growth = False
    batch_size = 2

    # Test G
    gen = Generator(final_resolution, prog_growth, device)
    print("init")

    z = torch.randn((batch_size, LATENT_DIM), device=device)
    x_fake = gen(z)
    print(x_fake.shape)

    # gen.synthesis_net.grow_new_block()
    # gen.synthesis_net.alpha = 0.5
    # x_fake = gen(z)
    # print(x_fake.shape)
    # gen.synthesis_net.fuse_new_block()
    # x_fake = gen(z)
    # print(x_fake.shape)
    
    # gen.synthesis_net.grow_new_block()
    # gen.synthesis_net.alpha = 0.5
    # x_fake = gen(z)
    # print(x_fake.shape)
    # gen.synthesis_net.fuse_new_block()
    # x_fake = gen(z)
    # print(x_fake.shape)


    # # Test D
    # dis = Discriminator(final_resolution, prog_growth, device)
    # print("init")
    # x_real = torch.zeros((batch_size, 3, final_resolution, final_resolution), device=device)
    # pred = dis(x_real)
    # print(pred.shape)

    # dis.grow_new_block()
    # dis.alpha = 0.5
    # x_real = F.interpolate(x_real, scale_factor=2)
    # pred = dis(x_real)
    # print(pred.shape)
    # dis.fuse_new_block()
    # pred = dis(x_real)
    # print(pred.shape)

    # dis.grow_new_block()
    # dis.alpha = 0.5
    # x_real = F.interpolate(x_real, scale_factor=2)
    # pred = dis(x_real)
    # print(pred.shape)
    # dis.fuse_new_block()
    # pred = dis(x_real)
    # print(pred.shape)