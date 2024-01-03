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
NUM_IMG_CHANNELS = 1
MIN_WORKING_RESOLUTION = 8   # Used during progressive growing



# ---
# Discriminator module

class Discriminator(nn.Module):

    def __init__(self, final_resolution=1024, prog_growth=False):

        super().__init__()

        self.prog_growth = prog_growth
        if prog_growth:
            self.working_resolution = MIN_WORKING_RESOLUTION
            self.has_unfused_new_block = False
            self.new_block = None
            self.to_rgb_skip = None
            self.alpha = None   
        else:
            self.working_resolution = final_resolution

        resolutions = [2**i for i in range(2, 12) if 2**i <= self.working_resolution]
        
        # Network body
        self.body = nn.ModuleList()
        self.body += [DiscriminatorBlock(MAX_CHANNELS, MAX_CHANNELS, is_last_block=True)]  # Last block, corresponding to lowest res (4x4)
        in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        for res in resolutions[1:]: # Blocks laid out in reverse order - i.e. lo-res to hi-res
            if res > 32: in_channels, out_channels = in_channels // 2, in_channels
            self.body += [DiscriminatorBlock(in_channels, out_channels)]
        self.body = self.body[::-1]

        # fromRGB layer at the output resolution
        self.from_rgb = Conv2d(NUM_IMG_CHANNELS, in_channels, kernel_size=1)

        # Last layer
        self.fc_layer = Linear(MAX_CHANNELS, 1, bias_init=0.0)

    def forward(self, image):
        x = self._compute_input_features(image)
        for block in self.body:
            x = block(x)
        pred = self.fc_layer(torch.reshape(x, (x.shape[0], -1)))
        return pred
    
    def _compute_input_features(self, image):
        if self.prog_growth:
            if self.has_unfused_new_block:
                x_main = self.from_rgb(image)
                x_main = self.new_block(x_main)
                x_skip = F.avg_pool2d(image, kernel_size=2, stride=2)
                x_skip = self.from_rgb_skip(x_skip)
                x = (1 - self.alpha) * x_skip + self.alpha * x_main
            else:
                x = self.from_rgb(image)
        else:
            x = self.from_rgb(image)
        return x
    
    def set_alpha(self, alpha):
        self.alpha = alpha

    def grow_new_block(self):

        # Double the working resolution
        self.working_resolution *= 2

        # Grow a new block at working resolution
        if self.working_resolution > 32:
            in_channels, out_channels = self.body[0].in_channels // 2, self.body[0].in_channels
        else:
            in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        self.new_block = DiscriminatorBlock(in_channels, out_channels).to(self.body[0].device)

        # Move the existing fromRGB layer to the skip connection route
        self.from_rgb_skip = Conv2d(NUM_IMG_CHANNELS, self.from_rgb.out_channels, kernel_size=1).to(self.body[0].device)
        with torch.no_grad():
            self.from_rgb_skip.weight.copy_(self.from_rgb.weight)
            self.from_rgb_skip.bias.copy_(self.from_rgb.bias)

        # And replace it with a newly initialized layer in the working resolution route
        self.from_rgb = Conv2d(NUM_IMG_CHANNELS, in_channels, kernel_size=1).to(self.body[0].device)

        # Reset alpha
        self.alpha = 0
        
        # Set flag
        self.has_unfused_new_block = True
    
    def fuse_new_block(self):
        
        # Fuse into the body
        self.body.insert(0, self.new_block)
        
        # Cleanup
        self.new_block = None
        self.from_rgb_skip = None
        self.alpha = None
        
        # Clear flag
        self.has_unfused_new_block = False
    

class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_last_block=False):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if is_last_block:
            layers = [MinibatchSDLayer(),
                      Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
                      nn.LeakyReLU(0.2),
                      Conv2d(in_channels, out_channels, kernel_size=4),
                      nn.LeakyReLU(0.2)]
        else:
            layers = [Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                      nn.LeakyReLU(0.2),
                      Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                      nn.LeakyReLU(0.2),
                      nn.AvgPool2d(kernel_size=2, stride=2)]

        self.block = nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):
        return self.block(x)
    

class MinibatchSDLayer(nn.Module):

    def forward(self, x):
        batch_size, _, height, width = x.shape
        sd = torch.mean(torch.sqrt(torch.var(x, dim=0, unbiased=False) + 1e-8))
        sd = sd.repeat((batch_size, 1, height, width))
        x = torch.cat([x, sd], dim=1)
        return x



# ---
# Progressive GAN generator

class ProGANGenerator(nn.Module):

    def __init__(self, final_resolution, prog_growth):

        super().__init__()

        self.prog_growth = prog_growth
        if prog_growth:
            self.working_resolution = MIN_WORKING_RESOLUTION
            self.has_unfused_new_block = False
            self.new_block = None
            self.to_rgb_skip = None
            self.alpha = None            
        else:
            self.working_resolution = final_resolution
        
        resolutions = [2**i for i in range(2, 12) if 2**i <= self.working_resolution]
        
        # Network body
        self.body = nn.ModuleList()
        self.body += [ProGANGeneratorBlock(MAX_CHANNELS, MAX_CHANNELS, is_first_block=True)]
        in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        for res in resolutions[1:]:
            if res > 32: in_channels, out_channels = out_channels, out_channels // 2
            self.body += [ProGANGeneratorBlock(in_channels, out_channels)]
        
        # toRGB layer at the output resolution
        self.to_rgb = Conv2d(out_channels, NUM_IMG_CHANNELS, kernel_size=1)
        
    def forward(self, z):
        batch_size = z.shape[0]
        z = z / (z.square().mean(dim=1, keepdim=True) + 1e-8).sqrt()  # Normalize
        x = z.reshape(batch_size, LATENT_DIM, 1, 1)
        for block in self.body:
            x = block(x)
        image = self._compute_output_image(x)
        return image

    def _compute_output_image(self, x):
        if self.prog_growth:
            if self.has_unfused_new_block:
                image_main = self.new_block(x)
                image_main = self.to_rgb(image_main)
                image_skip = self.to_rgb_skip(x)
                image_skip = F.interpolate(image_skip, scale_factor=2, mode='nearest')
                image = (1 - self.alpha) * image_skip + self.alpha * image_main
            else:
                image = self.to_rgb(x)
        else:
            image = self.to_rgb(x)
        return image
    
    def set_alpha(self, alpha):
        self.alpha = alpha

    def grow_new_block(self):

        # Double the working resolution
        self.working_resolution *= 2

        # Grow a new block at working resolution
        if self.working_resolution > 32:
            in_channels, out_channels = self.body[-1].out_channels, self.body[-1].out_channels // 2
        else:
            in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        self.new_block = ProGANGeneratorBlock(in_channels, out_channels).to(self.body[-1].device)

        # Move the existing toRGB layer to the skip connection route
        self.to_rgb_skip = Conv2d(self.to_rgb.in_channels, NUM_IMG_CHANNELS, kernel_size=1).to(self.body[-1].device)
        with torch.no_grad():
            self.to_rgb_skip.weight.copy_(self.to_rgb.weight)
            self.to_rgb_skip.bias.copy_(self.to_rgb.bias)

        # And replace it with a newly initialized layer in the working resolution route
        self.to_rgb = Conv2d(out_channels, NUM_IMG_CHANNELS, kernel_size=1).to(self.body[-1].device)
        
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


class ProGANGeneratorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_first_block=False):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        layers = []

        # 1st conv sub-block
        if is_first_block:
            layers += [ConvTranspose2d(in_channels, out_channels, kernel_size=4)]
        else:
            layers += [nn.Upsample(scale_factor=2, mode='nearest'), 
                       Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        layers += [nn.LeakyReLU(0.2)]
        
        # 2nd conv sub-block
        layers += [Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                   nn.LeakyReLU(0.2)]
        
        self.block = nn.Sequential(*layers)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):
        return self.block(x)



# ---
# StyleGAN generator

class StyleGANGenerator(nn.Module):

    def __init__(self, final_resolution=1024, prog_growth=False):
        super().__init__()
        self.prog_growth = prog_growth
        self.mapping_net = MappingNetwork()
        self.synthesis_net = SynthesisNetwork(final_resolution, prog_growth)
        
    def forward(self, z):
        z = z / (z.square().mean(dim=1, keepdim=True) + 1e-8).sqrt()  # Normalize
        w = self.mapping_net(z)
        image = self.synthesis_net(w)
        return image
    
    def set_alpha(self, alpha):
        self.synthesis_net.set_alpha(alpha)

    def grow_new_block(self):
        self.synthesis_net.grow_new_block()

    def fuse_new_block(self):
        self.synthesis_net.fuse_new_block()    


class MappingNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        layers = []
        for _ in range(MAPPING_NET_DEPTH):
            layers += [Linear(LATENT_DIM, LATENT_DIM, bias_init=0.0, lr_multiplier=0.01),
                       nn.LeakyReLU(0.2)]
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class SynthesisNetwork(nn.Module):

    def __init__(self, final_resolution, prog_growth):

        super().__init__()
        
        self.prog_growth = prog_growth
        if prog_growth:
            self.working_resolution = MIN_WORKING_RESOLUTION
            self.has_unfused_new_block = False
            self.new_block = None
            self.to_rgb_skip = None
            self.alpha = None            
        else:
            self.working_resolution = final_resolution
        
        resolutions = [2**i for i in range(2, 12) if 2**i <= self.working_resolution]
        
        # Network body
        self.body = nn.ModuleList()
        self.body += [SynthesisNetworkBlock(MAX_CHANNELS, MAX_CHANNELS, is_first_block=True)]
        in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        for res in resolutions[1:]:
            if res > 32: in_channels, out_channels = out_channels, out_channels // 2
            self.body += [SynthesisNetworkBlock(in_channels, out_channels)]
        
        # toRGB layer at the output resolution
        self.to_rgb = Conv2d(out_channels, NUM_IMG_CHANNELS, kernel_size=1)
        
        # Initial lowest res (4x4) feature map
        self.x_init = Parameter(torch.ones((1, MAX_CHANNELS, 4, 4)))

    def forward(self, w):
        batch_size, _ = w.shape
        x = torch.repeat_interleave(self.x_init, repeats=batch_size, dim=0)
        for block in self.body:
            x = block(x, w)
        image = self._compute_output_image(x, w)
        return image

    def _compute_output_image(self, x, w):
        if self.prog_growth:
            if self.has_unfused_new_block:
                image_main = self.new_block(x, w)
                image_main = self.to_rgb(image_main)
                image_skip = self.to_rgb_skip(x)
                image_skip = F.interpolate(image_skip, scale_factor=2, mode='nearest')
                image = (1 - self.alpha) * image_skip + self.alpha * image_main
            else:
                image = self.to_rgb(x)
        else:
            image = self.to_rgb(x)
        return image
    
    def set_alpha(self, alpha):
        self.alpha = alpha

    def grow_new_block(self):

        # Double the working resolution
        self.working_resolution *= 2

        # Grow a new block at working resolution
        if self.working_resolution > 32:
            in_channels, out_channels = self.body[-1].out_channels, self.body[-1].out_channels // 2
        else:
            in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        self.new_block = SynthesisNetworkBlock(in_channels, out_channels).to(self.body[-1].device)

        # Move the existing toRGB to the skip connection route
        self.to_rgb_skip = Conv2d(self.to_rgb.in_channels, NUM_IMG_CHANNELS, kernel_size=1).to(self.body[-1].device)
        with torch.no_grad():
            self.to_rgb_skip.weight.copy_(self.to_rgb.weight)
            self.to_rgb_skip.bias.copy_(self.to_rgb.bias)

        # And replace it with a newly initialized layer in the working resolution route
        self.to_rgb = Conv2d(out_channels, NUM_IMG_CHANNELS, kernel_size=1).to(self.body[-1].device)
        
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

    def __init__(self, in_channels, out_channels, is_first_block=False):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layers = nn.ModuleList()

        # 1st conv sub-block
        if not is_first_block:
            self.layers += [nn.Upsample(scale_factor=2, mode='nearest'),
                            Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        self.layers += [NoiseLayer(out_channels),
                        AdaINLayer(out_channels),
                        nn.LeakyReLU(0.2)]
        
        # 2nd conv sub-block
        self.layers += [Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        NoiseLayer(out_channels),
                        AdaINLayer(out_channels),
                        nn.LeakyReLU(0.2)]
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x, w):
        for layer in self.layers:
            if isinstance(layer, AdaINLayer): x = layer(x, w)
            else:                             x = layer(x)
        return x
    

class NoiseLayer(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        self.scaling_factors = Parameter(torch.zeros((1, num_channels, 1, 1)))

    def forward(self, x):
        batch_size, _, height, width = x.shape
        noise = torch.randn((batch_size, 1, height, width), device=x.device)
        return x + self.scaling_factors * noise


class AdaINLayer(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_channels, affine=False)
        self.affine_s = Linear(LATENT_DIM, num_channels, bias_init=1.0)
        self.affine_b = Linear(LATENT_DIM, num_channels, bias_init=0.0)

    def forward(self, x, w):
        batch_size, num_channels, _, _ = x.shape
        style_s = self.affine_s(w).reshape(batch_size, num_channels, 1, 1)
        style_b = self.affine_b(w).reshape(batch_size, num_channels, 1, 1)
        x = self.instance_norm(x) * style_s + style_b
        return x



# ---
# Custom primitive layers

class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias_init, lr_multiplier=1.0):
        super().__init__()
        self.weight = Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = Parameter(torch.full([out_features], fill_value=bias_init))
        self.weight_gain = lr_multiplier * (np.sqrt(2) / np.sqrt(in_features))  # For lr equalization + lr reduction
        self.bias_gain = lr_multiplier                                          #
    
    def forward(self, x):
        weight = self.weight * self.weight_gain
        bias = self.bias * self.bias_gain
        return F.linear(x, weight, bias)


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, padding_mode='reflect'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = Parameter(torch.full([out_channels], fill_value=0.0))
        self.weight_gain = np.sqrt(2) / np.sqrt(in_channels * kernel_size * kernel_size)  # For lr equalization
        self.padding = padding
        self.padding_mode = padding_mode

    def forward(self, x):
        weight = self.weight * self.weight_gain
        bias = self.bias
        x = F.pad(x, pad=[self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode)        
        return F.conv2d(x, weight, bias)
    

class ConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = Parameter(torch.full([out_channels], fill_value=0.0))
        self.weight_gain = 1 / np.sqrt(in_channels * kernel_size ** 2)   # For lr equalization
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

    def forward(self, x):
        weight = self.weight * self.weight_gain
        bias = self.bias
        return F.conv_transpose2d(x, weight, bias, self.stride, self.padding, self.output_padding, dilation=self.dilation)



# ---
# Unit test
if __name__ == '__main__':
    
    device = 'cuda'
    final_resolution = 32
    prog_growth = False
    batch_size = 4

    # Test
    dis = Discriminator(final_resolution, prog_growth).to(device)
    if prog_growth:
        for _ in range(2):
            dis.grow_new_block()
            dis.fuse_new_block()

    for i, p in enumerate(dis.parameters()):
        print(i, p.shape)