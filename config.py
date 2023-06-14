project = 'stylegan'
run_name = 'trial'

device = 'cuda'

final_resolution = 128

data_root = "/exports/lkeb-hpc/csrao/datasets/FFHQ"
final_batch_size = 32   # Corresponding to final resolution
num_iters = 200000

prog_growth = True
num_images_per_growth_half_cycle = 800000  # if following paper values, this will be 800k

log_freq = 20
checkpoint_freq = 5000