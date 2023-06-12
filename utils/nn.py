import torch
import torch.nn as nn



LATENT_DIM = 512
MAPPING_NET_DEPTH = 8



class Generator(nn.Module):

    def __init__(self, final_resolution=256, prog_growth=False, device='cpu'):
        super().__init__()
        self.prog_growth = prog_growth
        self.mapping_net = MappingNetwork(MAPPING_NET_DEPTH).to(device)
        self.synthesis_net = SynthesisNetwork(final_resolution, prog_growth, device)
        
    def forward(self, z):
        z = z / (z**2 + 1e-8).sum().sqrt()
        w = self.mapping_net(z)
        x = self.synthesis_net(w)
        return x


class MappingNetwork(nn.Module):

    def __init__(self, mapping_net_depth):
        super().__init__()

        layers = []
        for _ in range(mapping_net_depth):
            layers.extend([
                nn.Linear(LATENT_DIM, LATENT_DIM),
                nn.LeakyReLU(0.2)
            ])
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class SynthesisNetwork(nn.Module):

    def __init__(self, final_resolution, prog_growth, device):

        super().__init__()
        
        self.prog_growth = prog_growth # TODO: implement
        self.growth_signal = None
        
        resolutions = [2**i for i in range(2, 12) if 2**i <= final_resolution]
        
        self.max_channels = 512
        self.blocks = [
            SynthesisConvBlock(self.max_channels, self.max_channels,is_first_block=True, device=device),
            SynthesisConvBlock(self.max_channels, self.max_channels, device=device)
            ]
        for res in resolutions[1:]:
            if res < 32:  in_channels, out_channels = self.max_channels, self.max_channels
            else:         in_channels, out_channels = out_channels, out_channels // 2
            self.blocks.extend([
                SynthesisConvBlock(in_channels, out_channels, upsampling=True, device=device),
                SynthesisConvBlock(out_channels, out_channels, device=device)
                ])
            
        self.to_rgb = nn.Sequential(nn.Conv2d(out_channels, 3, kernel_size=1, device=device), nn.Tanh())
        self.x_init = nn.parameter.Parameter(torch.ones((1, self.max_channels, 4, 4), device=device))

    def set_growth_signal(self, growth_signal):
        self.growth_signal = growth_signal

    def forward(self, w):
        x = torch.repeat_interleave(self.x_init, repeats=w.shape[0], dim=0)
        for block in self.blocks:
            x = block(x, w)
        x = self.to_rgb(x)
        return x


class SynthesisConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, upsampling=False, is_first_block=False, device='cpu'):
        
        super().__init__()
        
        self.layers = []
        if not is_first_block:
            if upsampling:
                self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))    
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', device=device))
        
        self.layers.extend([
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
        self.affine_layer_s = nn.Linear(in_features=LATENT_DIM, out_features=num_channels, device=device)
        self.affine_layer_b = nn.Linear(in_features=LATENT_DIM, out_features=num_channels, device=device)
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

        self.prog_growth = prog_growth
        self.growth_signal = None

        resolutions = [2**i for i in range(2, 12) if 2**i <= final_resolution]
        resolutions = resolutions[::-1]

        num_channels = 16 * (1024 // final_resolution)
        backbone_layers = [nn.Conv2d(3, num_channels, kernel_size=1)]  # From RGB

        for res in resolutions[:-1]:            
            backbone_layers.extend([
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                nn.Conv2d(num_channels, min(2 * num_channels, 512), kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=2, stride=2)
                ])
            num_channels = min(2 * num_channels, 512)        
        self.backbone = nn.Sequential(*backbone_layers).to(device)

        classifier_layers = [
            nn.Conv2d(num_channels + 1, num_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels, num_channels, kernel_size=4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels, 1, kernel_size=1)  # Linear layer implemented as 1x1 conv2d
            ]
        self.classifier = nn.Sequential(*classifier_layers).to(device)

    def set_growth_signal(self, growth_signal):
        self.growth_signal = growth_signal

    def forward(self, x):
        features = self.backbone(x)
        features = self.minibatch_stddev_layer(features)
        pred = self.classifier(features).squeeze(-1).squeeze(-1)
        return pred
    
    def minibatch_stddev_layer(self, features):
        std_map = torch.full(
            (features.shape[0], 1, features.shape[2], features.shape[3]), 
            fill_value=float(torch.std(features, dim=0, unbiased=False).mean()), 
            device=features.device
            )
        features = torch.cat([features, std_map], dim=1)
        return features
    


# ---
# Unit test
if __name__ == '__main__':
    
    device = 'cuda'
    final_resolution = 128
    batch_size = 1

    gen = Generator(final_resolution, device)
    dis = Discriminator(final_resolution, device)
    print("init")

    z = torch.zeros((batch_size, LATENT_DIM), device=device)
    x_fake = gen(z)
    # print(x_fake)
    # print(x_fake.shape)
    print(dis(x_fake))

    x_real = torch.zeros((batch_size, 3, final_resolution, final_resolution), device=device)
    pred = dis(x_real)
    # print(pred.shape)
    print(pred)