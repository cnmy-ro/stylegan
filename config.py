from pathlib import Path

project = 'stylegan'
run_name = 'trial'
training_output_dir = Path("/exports/lkeb-hpc/csrao/git-personal/stylegan/training_output/")

device = 'cuda'

final_resolution = 128

data_root = Path("/exports/lkeb-hpc/csrao/datasets/FFHQ")
num_iters = 200000
lowres_caching = False  # TODO: Maybe remove?

fixed_batch_size = 16   # Only applicable when not using progressive growth

prog_growth = True
num_images_per_growth_half_cycle = 500  # if following paper values, this will be 800k

log_freq = 100
checkpoint_freq = 5000  # iters