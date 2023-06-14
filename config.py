from pathlib import Path

project = 'stylegan'
run_name = 'trial'
training_output_dir = Path("/exports/lkeb-hpc/csrao/git-personal/stylegan/training_output/")

device = 'cuda'

final_resolution = 128

data_root = Path("/exports/lkeb-hpc/csrao/datasets/FFHQ")
final_batch_size = 16   # Corresponding to final resolution
num_iters = 200000
lores_caching = True

prog_growth = True
num_images_per_growth_half_cycle = 800000  # if following paper values, this will be 800k

log_freq = 100
checkpoint_freq = 5000  # iters