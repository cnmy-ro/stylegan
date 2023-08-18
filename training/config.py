from pathlib import Path



project = 'stylegan'
run_name = 'stylegan_trial_nogrow'
training_output_dir = Path("/exports/lkeb-hpc/csrao/git-personal/stylegan/training/training_output/")
device = 'cuda'
seed = 0

data_root = Path("/exports/lkeb-hpc/csrao/datasets/FFHQ")
num_training_images = 10e6  # Training time in terms of #real images to be processed

fixed_batch_size = 16   # Only applicable when not using progressive growth

generator_design = 'stylegan'  # Options: 'progan' or 'stylegan'

prog_growth = False
num_images_per_growth_half_cycle = 800000  # if following paper values, this will be 800k

r1_gamma = 10

log_freq = 100
checkpoint_freq = 5000  # iters