from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import PIL



DATASET_RESOLUTION = 128
MIN_WORKING_RESOLUTION = 8   # Used during progressive growing
WORKING_RESOLUTION_TO_BATCH_SIZE_MAPPING = {8: 256, 16: 128, 32: 64, 64: 32, 128: 16}


class FFHQ128x128Dataset(Dataset):
    """
    Dataset of 128x128 FFHQ thumbnails. For fast prototyping purpose.
    """
    def __init__(self, root, split, prog_growth, lores_caching=False, training_output_dir=None):
        super().__init__()
        self.root = root
        self.split = split
        self.prog_growth = prog_growth
        if prog_growth: self.working_resolution = MIN_WORKING_RESOLUTION
        else:           self.working_resolution = DATASET_RESOLUTION

        self.lores_caching = lores_caching
        self.lores_cache_path = training_output_dir / Path("lores_cache")
        self.lores_cache_path.mkdir(exist_ok=True)
    
    def __len__(self):
        if self.split == 'train':  return 60000  # First 60k images for train
        elif self.split == 'val':  return 10000  # Remaining 10k for val

    def __getitem__(self, idx):
        image = self.fetch_image(idx)
        image = np.asarray(image) / 255.
        image = image * 2 - 1  
        image = torch.tensor(image, dtype=torch.float).permute(2,0,1)
        return {'image': image}
    
    def fetch_image(self, idx):
        idx = int(idx)
        subdir = str(idx - idx % 1000).zfill(5)
        image_path = self.root / Path(f"thumbnails128x128/{subdir}/{str(idx).zfill(5)}.png")
        cached_image_path = self.lores_cache_path / Path(f"{self.working_resolution}x{self.working_resolution}/{subdir}/{str(idx).zfill(5)}.png")
        
        if not self.lores_caching or self.working_resolution == DATASET_RESOLUTION:
            image = PIL.Image.open(image_path)
            if self.working_resolution < DATASET_RESOLUTION:
                image = image.resize((self.working_resolution, self.working_resolution), PIL.Image.BOX)
        else:            
            if cached_image_path.exists():
                image = PIL.Image.open(cached_image_path)
            else:
                image = PIL.Image.open(image_path)
                image = image.resize((self.working_resolution, self.working_resolution), PIL.Image.BOX)
                cached_image_path.parents[0].mkdir(parents=True, exist_ok=True)
                image.save(cached_image_path)
    
        return image
    
    def double_working_resolution(self):
        self.working_resolution *= 2



class InfiniteSampler(Sampler):
    
    def __init__(self, dataset_size, split='train'):
        self.dataset_size = dataset_size
        self.split = split

    def __iter__(self):
        while True:
            if self.split == 'train':
                yield from torch.randint(low=0, high=self.dataset_size, size=(1,))
            elif self.split == 'val':
                yield from torch.randint(low=60000, high=60000 + self.dataset_size, size=(1,))