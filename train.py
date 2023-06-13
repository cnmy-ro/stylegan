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



def log_to_dashboard(loss_g, loss_d, fake, iter_counter):
    if iter_counter % config.log_freq == 0:
        samples_grid = make_grid(fake.detach().cpu(), nrow=8, normalize=True, value_range=(-1,1)).permute((1,2,0)).numpy()
        log_dict = {
            'Loss: G': float(loss_g.detach().cpu()), 
            'Loss: D': float(loss_d.detach().cpu()),
            'Samples': [wandb.Image(samples_grid)]
            }
        wandb.log(log_dict, step=iter_counter)


def grow_model(generator, discriminator, dataset, iter_counter):

    def iter_at_start_of_growth_cycle(iter_counter):
        return iter_counter % config.num_iters_growth_cycle == 0
    
    def iter_in_fading_phase(iter_counter):
        return (iter_counter // config.num_iters_growth_cycle) % 2 != 0
        
    # If already at max resolution, return
    if discriminator.output_resolution == FFHQ128x128Dataset.DATASET_RESOLUTION:
        return
        
    # If at a growth iter, grow
    if iter_at_start_of_growth_cycle(iter_counter):

        # If at the start of the fading-in phase, grow blocks and set alpha=0
        if iter_in_fading_phase(iter_counter):

            generator.synthesis_net.grow_new_block()
            discriminator.grow_new_block()

            generator.synthesis_net.alpha = 0
            discriminator.alpha = 0

            dataset.double_output_resolution()

        # If at the start of the stabilization phase, fuse blocks into net body
        else:
            generator.synthesis_net.fuse_grown_block()
            discriminator.fuse_grown_block()

    # If inside a fading-in phase, update alpha
    elif iter_in_fading_phase(iter_counter):
        prev_growth_iter = (iter_counter // config.num_iters_growth_cycle) * config.num_iters_growth_cycle
        alpha = (iter_counter - prev_growth_iter) / config.num_iters_growth_cycle
        generator.synthesis_net.alpha = alpha
        discriminator.alpha = alpha


    
def main():
    
    # Data
    dataset = FFHQ128x128Dataset(config.data_root, 'train', config.prog_growth)
    sampler = InfiniteSampler(dataset_size=len(dataset), shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    
    # Model
    generator = Generator(config.final_resolution, config.prog_growth, config.device)
    discriminator = Discriminator(config.final_resolution, config.prog_growth, config.device)

    # Optimizers
    opt_g = Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.99))
    opt_d = Adam(discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))    
    
    # Dashboard
    wandb.init(project=config.project, name=config.run_name)
    
    # Training loop
    for iter_counter in range(1, config.num_iters + 1):

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

        # Progressive growth
        if config.prog_growth:
            grow_model(None, None, None, iter_counter)

        # Log
        log_to_dashboard(loss_g, loss_d, fake, iter_counter)


# ---
# Run
if __name__ == '__main__':
    main()