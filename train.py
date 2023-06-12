import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
import wandb
from tqdm import tqdm

import config
from utils.nn import Generator, Discriminator, LATENT_DIM
from utils.dataset import FFHQ128x128Dataset, InfiniteSampler
from utils.losses import nsgan_criterion



def log_to_dashboard(loss_g, loss_d, fake, iteration):
    samples_grid = make_grid(fake.detach().cpu(), nrow=8, normalize=True, value_range=(-1,1)).permute((1,2,0)).numpy()
    log_dict = {
        'Loss: G': float(loss_g.detach().cpu()), 
        'Loss: D': float(loss_d.detach().cpu()),
        'Samples': [wandb.Image(samples_grid)]
        }
    wandb.log(log_dict, step=iteration)


def main():
    
    dataset = FFHQ128x128Dataset(config.data_root, split='train')
    sampler = InfiniteSampler(size=len(dataset), shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    
    generator = Generator(config.output_resolution, config.device)
    discriminator = Discriminator(config.output_resolution, config.device)

    opt_g = Adam(generator.parameters(), lr=config.lr_g)
    opt_d = Adam(discriminator.parameters(), lr=config.lr_g)    
   
    wandb.init(project=config.wandb_project, name=config.wandb_run_name)
    
    for it in tqdm(range(config.num_iters)):
        
        # Update G
        opt_g.zero_grad(set_to_none=True)
        for p in discriminator.parameters(): p.requires_grad = False
        latent = torch.randn((config.batch_size, LATENT_DIM), device=config.device)
        fake = generator(latent)
        loss_g = nsgan_criterion(discriminator(fake), is_real=True)
        loss_g.backward()
        opt_g.step()

        # Update D
        for p in discriminator.parameters(): p.requires_grad = True
        opt_d.zero_grad(set_to_none=True)
        batch = next(iter(dataloader))
        real = batch['image'].to(config.device)
        loss_d = nsgan_criterion(discriminator(real), is_real=True) + nsgan_criterion(discriminator(fake.detach()), is_real=False)
        loss_d.backward()
        opt_d.step()

        # Log
        if it % config.log_freq == 0:
            log_to_dashboard(loss_g, loss_d, fake, it)


# ---
# Run
if __name__ == '__main__':
    main()