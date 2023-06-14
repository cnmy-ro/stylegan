import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import PIL



DATASET_RESOLUTION = 128
MIN_WORKING_RESOLUTION = 8   # Used during progressive growing



class FFHQ128x128Dataset(Dataset):
    """
    Dataset of 128x128 FFHQ thumbnails. For quick model testing purposes.
    """
    def __init__(self, root, split, prog_growth):
        super().__init__()
        self.root = root
        self.split = split
        self.prog_growth = prog_growth
        if prog_growth: self.working_resolution = MIN_WORKING_RESOLUTION
        else:           self.working_resolution = DATASET_RESOLUTION
    
    def __len__(self):
        if self.split == 'train':  return 60000
        elif self.split == 'val':  return 10000    

    def __getitem__(self, idx):
        
        idx = int(idx)
        subdir = str(idx - idx % 1000).zfill(5)
        path = f"{self.root}/thumbnails128x128/{subdir}/{str(idx).zfill(5)}.png"
        image = PIL.Image.open(path)

        if self.working_resolution < 128:
            image = image.resize((self.working_resolution, self.working_resolution), PIL.Image.BOX)

        image = np.asarray(image) / 255.
        image = image * 2 - 1  
        image = torch.tensor(image, dtype=torch.float).permute(2,0,1)

        return {'image': image}

    def double_working_resolution(self):
        self.working_resolution *= 2



class InfiniteSampler(Sampler):
    
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size

    def __iter__(self):
        while True:
            yield from torch.randperm(self.dataset_size)