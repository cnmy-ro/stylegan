project = 'stylegan'
run_name = 'trial'

device = 'cuda'

final_resolution = 128

data_root = "/exports/lkeb-hpc/csrao/datasets/FFHQ"
final_batch_size = 32   # Corresponding to final resolution
num_iters = 450000

prog_growth = True
num_iters_growth_cycle = 50000  # if following paper values, this will be equivalent to 800k images

log_freq = 20