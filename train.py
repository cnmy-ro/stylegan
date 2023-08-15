from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam
from torchvision.utils import make_grid
import wandb

import config
from utils.dataset import FFHQ128x128Dataset, InfiniteDataLoader, DATASET_RESOLUTION, WORKING_RESOLUTION_TO_BATCH_SIZE
from utils.nn import StyleGANGenerator, ProGANGenerator, Discriminator, LATENT_DIM, MIN_WORKING_RESOLUTION
from utils.criteria import nsgan_loss, r1_regularizer


# ---
# Reproducibility
np.random.seed(config.seed)
torch.manual_seed(config.seed)


# ---
# Utils

def log_to_dashboard(loss_g, loss_d, real, fake, iter_counter, image_counter, max_samples=16):

    loss_g, loss_d = float(loss_g.detach().cpu()), float(loss_d.detach().cpu())

    if iter_counter % config.log_freq == 0:

        print(f"\n----- Iters: {iter_counter}  |  Images: {image_counter} -----")
        print(f"\t\t Loss G: {loss_g:.3f}")
        print(f"\t\t Loss D: {loss_d:.3f}\n")

        if real.shape[0] > max_samples: real = real[:max_samples]
        if fake.shape[0] > max_samples: fake = fake[:max_samples]        
        data_grid = make_grid(real.detach().cpu(), nrow=8, normalize=True, value_range=(-1, 1)).permute((1, 2, 0)).numpy()
        samples_grid = make_grid(fake.detach().cpu(), nrow=8, normalize=True, value_range=(-1, 1)).permute((1, 2, 0)).numpy()
        log_dict = {
            'Num reals processed': image_counter,
            'Loss: G': loss_g,
            'Loss: D': loss_d,
            'Samples': [wandb.Image(samples_grid)],
            'Data': [wandb.Image(data_grid)]
            }

        wandb.log(log_dict, step=iter_counter)


def dump_checkpoint(generator, discriminator, opt_g, opt_d, iter_counter, image_counter):
    
    if iter_counter % config.checkpoint_freq == 0:
        
        checkpoint = {
            'iter_counter': iter_counter,
            'image_counter': image_counter,
            'prog_growth': config.prog_growth, 
            'num_images_per_growth_half_cycle': config.num_images_per_growth_half_cycle,
            'net_g_state_dict': generator.state_dict(),
            'net_d_state_dict': discriminator.state_dict(),
            'opt_g_state_dict': opt_g.state_dict(),
            'opt_d_state_dict': opt_d.state_dict()
            }
        
        checkpoint_dir = Path(f"{config.training_output_dir}/checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        path = checkpoint_dir / Path(f"iter_{iter_counter}.pt")
        torch.save(checkpoint, path)


def grow_model(generator, discriminator, dataloader, image_counter):

    def at_start_of_growth_half_cycle():
        tol = dataloader.batch_size
        return image_counter % config.num_images_per_growth_half_cycle < tol
    
    def in_transition_phase():
        return (image_counter // config.num_images_per_growth_half_cycle) % 2 != 0
    
    def in_stabilization_phase():
        return (image_counter // config.num_images_per_growth_half_cycle) % 2 == 0

    def calc_alpha():
        prev_growth_milestone = (image_counter // config.num_images_per_growth_half_cycle) * config.num_images_per_growth_half_cycle        
        alpha = (image_counter - prev_growth_milestone) / config.num_images_per_growth_half_cycle        
        return alpha

    # If already at max resolution, do nothing and return
    if dataloader.dataset.working_resolution == DATASET_RESOLUTION:
        return generator, discriminator, dataloader
    
    # If at the start of a growth half-cycle, handle growth
    if at_start_of_growth_half_cycle():

        # If this is the start of transition phase, double the resolution and grow new block
        if in_transition_phase():

            generator.grow_new_block()
            discriminator.grow_new_block()

            dataloader.dataset.double_working_resolution()
            batch_size = WORKING_RESOLUTION_TO_BATCH_SIZE[dataloader.dataset.working_resolution]
            dataloader = InfiniteDataLoader(dataloader.dataset, batch_size=batch_size, num_workers=dataloader.num_workers, shuffle=True)
            # print("fading phase start")

        # If this is the start of stabilization phase, fuse the block into net body
        elif in_stabilization_phase():
            generator.fuse_new_block()
            discriminator.fuse_new_block()
            dataloader.dataset.reset_alpha()    
            # print("FUSED")

    # If inside the transition phase, update alpha
    elif in_transition_phase():
        alpha = calc_alpha()
        generator.set_alpha(alpha)
        discriminator.set_alpha(alpha)
        dataloader.dataset.set_alpha(alpha)
        # print("fading...", alpha)

    # If inside the stabilization phase, do nothing
    elif in_stabilization_phase():
        pass

    return generator, discriminator, dataloader


def main():
    
    # Config
    config.training_output_dir.mkdir(exist_ok=True)

    # Data
    dataset = FFHQ128x128Dataset(config.data_root, 'train', config.prog_growth)
    if config.prog_growth: init_batch_size = WORKING_RESOLUTION_TO_BATCH_SIZE[MIN_WORKING_RESOLUTION]
    else:                  init_batch_size = config.fixed_batch_size
    dataloader = InfiniteDataLoader(dataset, batch_size=init_batch_size, num_workers=1, shuffle=True)
    
    # Model
    if config.generator_design == 'progan':  generator_class = ProGANGenerator
    if config.generator_design == 'stylegan': generator_class = StyleGANGenerator
    generator = generator_class(DATASET_RESOLUTION, config.prog_growth).to(config.device)
    discriminator = Discriminator(DATASET_RESOLUTION, config.prog_growth).to(config.device)

    # Optimizers
    opt_g = Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.99), eps=1e-8)
    opt_d = Adam(discriminator.parameters(), lr=0.001, betas=(0.0, 0.99), eps=1e-8)
    
    # Dashboard    
    wandb.init(project=config.project, name=config.run_name, dir=f"{config.training_output_dir}")
    
    # Training loop
    iter_counter, image_counter = 0, 0
    print("Training started")
    while image_counter < config.num_training_images:
        
        # Update G
        opt_g.zero_grad(set_to_none=True)
        for p in discriminator.parameters(): p.requires_grad = False
        latent = torch.randn((dataloader.batch_size, LATENT_DIM), device=config.device)
        fake = generator(latent)
        loss_g = nsgan_loss(discriminator(fake), is_real=True)
        loss_g.backward()
        opt_g.step()

        # Update D
        for p in discriminator.parameters(): p.requires_grad = True
        opt_d.zero_grad(set_to_none=True)
        batch = next(iter(dataloader))
        real = batch['image'].to(config.device)
        loss_d = nsgan_loss(discriminator(real), is_real=True) + nsgan_loss(discriminator(fake.detach()), is_real=False)
        if config.r1_gamma > 0: loss_d += r1_regularizer(discriminator, real, config.r1_gamma)
        loss_d.backward()
        opt_d.step()

        # Log
        iter_counter += 1
        image_counter += real.shape[0]        
        log_to_dashboard(loss_g, loss_d, real, fake, iter_counter, image_counter)

        # Checkpoint
        dump_checkpoint(generator, discriminator, opt_g, opt_d, iter_counter, image_counter)

        # Progressive growing        
        if config.prog_growth:
            generator, discriminator, dataloader = grow_model(generator, discriminator, dataloader, image_counter)
    
    print("Training complete")



# ---
# Run
if __name__ == '__main__':
    main()