import torch
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
import numpy as np
import PIL



DATASET_RESOLUTION = 128



class FFHQ128x128Dataset(Dataset):
    """
    Dataset of 128x128 FFHQ thumbnails. For quick model testing purposes.
    """
    def __init__(self, root, split, prog_growth):
        super().__init__()
        self.root = root
        self.split = split
        self.prog_growth = prog_growth
        if prog_growth: self.output_resolution = 8
        else:           self.output_resolution = DATASET_RESOLUTION
    
    def __len__(self):
        if self.split == 'train':  return 60000
        elif self.split == 'val':  return 10000    

    def __getitem__(self, idx):
        
        idx = int(idx)
        subdir = str(idx - idx % 1000).zfill(5)
        path = f"{self.root}/thumbnails128x128/{subdir}/{str(idx).zfill(5)}.png"
        image = PIL.Image.open(path)

        image = np.asarray(image) / 255.
        image = image * 2 - 1
        
        image = torch.tensor(image, dtype=torch.float).permute(2,0,1)        
        if self.output_resolution < 128:
            image = F.interpolate(image, size=self.output_resolution, mode='bicubic')

        return {'image': image}

    def double_output_resolution(self):
        self.output_resolution *= 2



class InfiniteSampler(Sampler):
    
    def __init__(self, dataset_size, shuffle=True):
        self.dataset_size = dataset_size
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            if self.shuffle: yield from torch.randperm(self.dataset_size)
            else:            yield from torch.arange(self.dataset_size)