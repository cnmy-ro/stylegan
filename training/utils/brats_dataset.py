"""
Dataset class for BraTS 2020 training data downloaded from Kaggle: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


DATASET_RESOLUTION = 128
MIN_WORKING_RESOLUTION = 8   # Used during progressive growing
WORKING_RESOLUTION_TO_BATCH_SIZE = {8: 128, 16: 128, 32: 64, 64: 32, 128: 16}
DATA_SUBSETS_INFO_FILE = Path("/exports/lkeb-hpc/csrao/datasets/BraTS2020_train_kaggle/brats20_subsets_info.csv")


class BraTS20Dataset(Dataset):

    def __init__(self, root, prog_growth):
        
        super().__init__()        
    
        data_dir = Path(f"{root}/BraTS2020_training_data/content/data")
        subsets_info = pd.read_csv(DATA_SUBSETS_INFO_FILE)
        slice_filenames = list(subsets_info['filenames'])
        self.slice_filepaths = [f"{data_dir}/{filename}" for filename in slice_filenames]

        self.prog_growth = prog_growth
        if prog_growth: 
            self.working_resolution = MIN_WORKING_RESOLUTION
            self.alpha = None
        else:
            self.working_resolution = DATASET_RESOLUTION
    
    def __len__(self):
        return len(self.slice_filepaths)

    def __getitem__(self, idx):
        image = self._fetch_slice(idx)
        image = rescale_intensity(image, to_range=[-1, 1])
        example = {'image': image}
        return example
    
    def _fetch_slice(self, idx):

        with h5py.File(self.slice_filepaths[idx], 'r') as hf:
            slice_array = np.asarray(hf['image'])
        
        contrasts = {
            't1w':    slice_array[:, :, 1],
            't1w_gd': slice_array[:, :, 2],
            't2w':    slice_array[:, :, 3],
            'flair':  slice_array[:, :, 0]
            }
        contrast = np.random.choice(list(contrasts.keys()))
        image = contrasts[contrast]

        image = np.transpose(image, (1, 0))  # (W,H) to (H,W)        
        image = torch.from_numpy(image).to(torch.float).unsqueeze(0)  # To tensor of shape (C,H,W)        

        if not self.prog_growth:
            return F.resize(image, (DATASET_RESOLUTION, DATASET_RESOLUTION), F.InterpolationMode.BILINEAR)
        else:
            if self.working_resolution < DATASET_RESOLUTION:
                image_lowres_main = F.resize(image, (self.working_resolution, self.working_resolution), F.InterpolationMode.BILINEAR)
                if self.alpha is None:  # In stabilization phase, output only the lowres image
                    return image_lowres_main
                else:                   # In transition phase, alpha-blend between current lowres and a 2x nearest upsampled version of this image from previous resolution
                    assert self.working_resolution > MIN_WORKING_RESOLUTION
                    image_lowres_skip = F.resize(image, (self.working_resolution // 2, self.working_resolution // 2), F.InterpolationMode.BILINEAR)  # This mirrors what happens inside the generator during transition phase
                    image_lowres_skip = F.resize(image_lowres_skip, (self.working_resolution, self.working_resolution), F.InterpolationMode.NEAREST) #
                    return (1 - self.alpha) * image_lowres_skip + self.alpha * image_lowres_main                                           #
            else:
                return F.resize(image, (DATASET_RESOLUTION, DATASET_RESOLUTION), F.InterpolationMode.BILINEAR)

    def double_working_resolution(self):
        self.working_resolution *= 2
        self.alpha = 0
    
    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def reset_alpha(self):
        self.alpha = None
        

def rescale_intensity(image, from_range=None, to_range=[-1,1], clip=False):
    
    if from_range is None:
        if image.min() == image.max():
            return image
        from_range = (image.min(), image.max())

    # Clip values that are outside of the source range
    if clip:
        if isinstance(image, np.ndarray):
            image = np.clip(image, from_range[0], from_range[1])
        elif isinstance(image, torch.Tensor):
            image = torch.clip(image, from_range[0], from_range[1])

    # First rescale to [0, 1]
    image = (image - from_range[0]) / (from_range[1] - from_range[0])        
    
    # The rescale to target range
    image = image * (to_range[1] - to_range[0]) + to_range[0]

    return image