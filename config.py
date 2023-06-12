device = 'cuda'

output_resolution = 128


data_root = "/exports/lkeb-hpc/csrao/datasets/FFHQ"
batch_size = 32
num_iters = 200000
lr_g = 0.0001
lr_d = 0.0001

wandb_project = 'stylegan'
wandb_run_name = 'trial'
log_freq = 20