device = 'cuda'

final_resolution = 128


data_root = "/exports/lkeb-hpc/csrao/datasets/FFHQ"
final_batch_size = 32   # Corresponding to final resolution

num_iters = 100
lr_g = 0.0001
lr_d = 0.0001

prog_growth = True
num_iters_growth_cycle = 10  # if following paper values, this will be equivalent to 800k images


wandb_project = 'stylegan'
wandb_run_name = 'trial'
log_freq = 20