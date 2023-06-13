import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



LATENT_DIM = 512
MAPPING_NET_DEPTH = 8
MAX_CHANNELS = 512


class Generator(nn.Module):

    def __init__(self, final_resolution=256, prog_growth=False, device='cpu'):
        super().__init__()
        self.prog_growth = prog_growth
        self.mapping_net = MappingNetwork(MAPPING_NET_DEPTH, device)
        self.synthesis_net = SynthesisNetwork(final_resolution, prog_growth, device)
        
    def forward(self, z):
        z = z / (z**2 + 1e-8).sum().sqrt()
        w = self.mapping_net(z)
        x = self.synthesis_net(w)
        return x


class MappingNetwork(nn.Module):

    def __init__(self, mapping_net_depth, device):

        super().__init__()

        layers = []
        for _ in range(mapping_net_depth):
            layers.extend([
                Linear(bias_init_val=0.0, in_features=LATENT_DIM, out_features=LATENT_DIM),
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
            output_resolution = 8
            self.has_unfused_new_block = False
            self.new_block = None
            self.to_rgb_skip = None
            self.alpha = None            
        else:
            output_resolution = final_resolution
        
        resolutions = [2**i for i in range(2, 12) if 2**i <= output_resolution]
        
        # Network body
        self.body = [SynthesisNetworkBlock(MAX_CHANNELS, MAX_CHANNELS, is_first_block=True, device=device)]
        in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        for res in resolutions[1:]:
            if res > 32: in_channels, out_channels = out_channels, out_channels // 2
            self.body.append(SynthesisNetworkBlock(in_channels, out_channels, device=device))
        
        # toRGB layer at the working resolution
        self.to_rgb = Conv2d(out_channels, 3, kernel_size=1, device=device)
        
        # Initial lowest res (4x4) feature map
        self.x_init = nn.parameter.Parameter(torch.ones((1, MAX_CHANNELS, 4, 4), device=device))

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
    
    def grow_new_block(self):

        # Grow a new block at 2x res
        in_channels, out_channels = self.body[-1].out_channels, self.body[-1].out_channels // 2
        self.new_block = SynthesisNetworkBlock(in_channels, out_channels, device=self.device)

        # Replace current toRGB layer with 2x res one
        self.to_rgb = Conv2d(out_channels, 3, kernel_size=1, device=device)
        
        # Add a 2x res toRGB skip layer
        self.to_rgb_skip = Conv2d(in_channels, 3, kernel_size=1, device=device)
        
        # Flag
        self.has_unfused_new_block = True
    
    def fuse_new_block(self):
        
        # Fuse into the body
        self.body.append(self.new_block)
        
        # Cleanup
        self.new_block = None
        self.to_rgb_skip = None
        self.alpha = None
        
        # Flag
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
            self.layers.append(Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', device=device))
        self.layers.extend([
            NoiseLayer(out_channels, device),
            AdaINLayer(out_channels, device),
            nn.LeakyReLU(0.2)
            ])
        
        # 2nd conv block
        self.layers.extend([
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', device=device),
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
        self.scaling_factors = nn.parameter.Parameter(torch.zeros((1, num_channels, 1, 1), device=device))

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        noise = torch.randn((batch_size, 1, height, width), device=x.device)
        return x + self.scaling_factors * noise


class AdaINLayer(nn.Module):
    
    def __init__(self, num_channels, device):
        super().__init__()
        self.affine_layer_s = Linear(bias_init_val=1.0, in_features=LATENT_DIM, out_features=num_channels, device=device)
        self.affine_layer_b = Linear(bias_init_val=0.0, in_features=LATENT_DIM, out_features=num_channels, device=device)
        self.eps = 1e-5

    def forward(self, x, w):
        style_s = self.affine_layer_s(w).unsqueeze(-1).unsqueeze(-1)
        style_b = self.affine_layer_b(w).unsqueeze(-1).unsqueeze(-1)
        mean, var = x.mean(dim=[-1, -2], keepdims=True), x.var(dim=[-1, -2], keepdims=True, unbiased=False) + self.eps
        x = style_s * ((x - mean) / var.sqrt()) + style_b
        return x


class Discriminator(nn.Module):

    def __init__(self, final_resolution=256, prog_growth=False, device='cpu'):

        super().__init__()

        self.device = device

        self.prog_growth = prog_growth
        if prog_growth:
            output_resolution = 8
            self.has_unfused_new_block = False
            self.new_block = None
            self.to_rgb_skip = None
            self.alpha = None   
        else:
            output_resolution = final_resolution

        resolutions = [2**i for i in range(2, 12) if 2**i <= output_resolution]
        
        # Network body
        self.body = [DiscriminatorBlock(MAX_CHANNELS, MAX_CHANNELS, is_last_block=True, device=device)]  # Last block, corresponding to lowest res (4x4)
        in_channels, out_channels = MAX_CHANNELS, MAX_CHANNELS
        for res in resolutions[1:]: # Blocks laid out in reverse order - i.e. lo-res to hi-res
            if res > 32: in_channels, out_channels = in_channels // 2, in_channels
            self.body.append(DiscriminatorBlock(in_channels, out_channels, device=device))
        self.body = self.body[::-1]

        # fromRGB layer at the working resolution
        self.from_rgb = Conv2d(3, in_channels, kernel_size=1, device=device)

        # Last layer
        self.classifier = Linear(bias_init_val=0.0, in_features=MAX_CHANNELS, out_features=1, device=device)

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
                x_skip = F.interpolate(x, size=(x.shape[2] // 2, x.shape[3] // 2), mode='nearest')
                x_skip = self.from_rgb_skip(x_skip)
                x = (1 - self.alpha) * x_skip + self.alpha * x_main
            else:
                x = self.from_rgb(x)
        else:
            x = self.from_rgb(x)
        return x
    
    def grow_new_block(self):

        # Grow a new block at 2x res
        in_channels, out_channels = self.body[0].in_channels // 2, self.body[0].in_channels
        self.new_block = DiscriminatorBlock(in_channels, out_channels, device=self.device)

        # Replace current fromRGB layer with 2x res one
        self.from_rgb = Conv2d(3, in_channels, kernel_size=1, device=device)
        
        # Add a 2x res fromRGB skip layer
        self.from_rgb_skip = Conv2d(3, out_channels, kernel_size=1, device=device)
        
        # Flag
        self.has_unfused_new_block = True
    
    def fuse_new_block(self):
        
        # Fuse into the body
        self.body.insert(0, self.new_block)
        
        # Cleanup
        self.new_block = None
        self.to_rgb_skip = None
        self.alpha = None
        
        # Flag
        self.has_unfused_new_block = False
    
    

class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_last_block=False, device='cpu'):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if is_last_block:
            layers = [
                MinibatchSDLayer(),
                Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                Conv2d(in_channels, out_channels, kernel_size=4),
                nn.LeakyReLU(0.2)
            ]            
        else:
            layers= [
                Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=2, stride=2)  # TODO: check which subsampling op used in paper
                ]

        self.block = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.block(x)



# ---
# Custom layers and layers with custom initialization

class Linear(nn.Linear):
    def __init__(self, bias_init_val, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.init.normal_(self.weight)
        self.bias = nn.init.constant_(self.bias, val=bias_init_val)

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.init.normal_(self.weight)
        self.bias = nn.init.constant_(self.bias, val=0.0)

class MinibatchSDLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        std_map = torch.full(
            (x.shape[0], 1, x.shape[2], x.shape[3]), 
            fill_value=float(torch.std(x, dim=0, unbiased=False).mean()), 
            device=x.device
            )
        x = torch.cat([x, std_map], dim=1)
        return x



# ---
# Unit test
if __name__ == '__main__':
    
    device = 'cuda'
    final_resolution = 4
    prog_growth = True
    batch_size = 1

    # Test G
    # gen = Generator(final_resolution, prog_growth, device)
    print("init")

    # z = torch.zeros((batch_size, LATENT_DIM), device=device)
    # x_fake = gen(z)
    # print(x_fake.shape)

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


    # Test D
    dis = Discriminator(final_resolution, prog_growth, device)
    print("init")
    x_real = torch.zeros((batch_size, 3, final_resolution, final_resolution), device=device)
    pred = dis(x_real)
    # # print(pred.shape)
    # print(pred)

    dis.grow_new_block()
    dis.alpha = 0.5
    x_real = F.interpolate(x_real, scale_factor=2)
    pred = dis(x_real)
    print(pred.shape)
    dis.fuse_new_block()
    pred = dis(x_real)
    print(pred.shape)

    dis.grow_new_block()
    dis.alpha = 0.5
    x_real = F.interpolate(x_real, scale_factor=2)
    pred = dis(x_real)
    print(pred.shape)
    dis.fuse_new_block()
    pred = dis(x_real)
    print(pred.shape)