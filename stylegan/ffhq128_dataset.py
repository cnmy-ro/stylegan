from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import PIL


DATASET_RESOLUTION = 128
MIN_WORKING_RESOLUTION = 8   # Used during progressive growing
WORKING_RESOLUTION_TO_BATCH_SIZE = {8: 128, 16: 128, 32: 64, 64: 32, 128: 16}


class FFHQ128Dataset(Dataset):
    """
    Dataset of 128x128 FFHQ thumbnails. For fast prototyping purpose.
    """
    def __init__(self, root, split, prog_growth):
        super().__init__()
        self.root = root
        self.split = split
        self.prog_growth = prog_growth
        if prog_growth: 
            self.working_resolution = MIN_WORKING_RESOLUTION
            self.alpha = None
        else:
            self.working_resolution = DATASET_RESOLUTION
    
    def __len__(self):
        if self.split == 'train':  return 60000  # First 60k images for train
        elif self.split == 'val':  return 10000  # Remaining 10k for val

    def __getitem__(self, idx):
        if self.split == 'val': idx += 60000
        image = self.fetch_image(idx)
        image = (image / 255.) * 2 - 1  
        image = torch.tensor(image, dtype=torch.float).permute(2,0,1)
        return {'image': image}
    
    def fetch_image(self, idx):
        idx = int(idx)
        subdir = str(idx - idx % 1000).zfill(5)
        image_path = self.root / Path(f"thumbnails128x128/{subdir}/{str(idx).zfill(5)}.png")
        image_orig = PIL.Image.open(image_path)
        
        if not self.prog_growth:
            return np.asarray(image_orig)
        else:
            image_lowres_main = image_orig.resize((self.working_resolution, self.working_resolution), PIL.Image.BICUBIC)                
            if self.alpha is None:  # In stabilization phase, output only the lowres image
                return np.asarray(image_lowres_main)
            else:                   # In transition phase, alpha-blend between current lowres and a 2x nearest upsampled version of this image from previous resolution
                image_lowres_skip = image_orig.resize((self.working_resolution // 2, self.working_resolution // 2), PIL.Image.BICUBIC) # This mirrors what happens inside the generator during transition phase
                image_lowres_skip = image_lowres_skip.resize((self.working_resolution, self.working_resolution), PIL.Image.NEAREST)    #
                return (1 - self.alpha) * np.asarray(image_lowres_skip) + self.alpha * np.asarray(image_lowres_main)                   #
    
    def double_working_resolution(self):
        self.working_resolution *= 2
        self.alpha = 0
    
    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def reset_alpha(self):
        self.alpha = None

class InfiniteDataLoader(DataLoader):
    """
    Taken from:  https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch