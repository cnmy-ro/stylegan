import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import PIL



class FFHQ128x128Dataset(Dataset):
    """
    Dataset of 128x128 FFHQ thumbnails. For quick model testing purposes.
    """
    def __init__(self, root, split, prog_growth):
        super().__init__()
        self.root = root
        self.split = split
        self.prog_growth = prog_growth
        self.growth_signal = None
    
    def __len__(self):
        if self.split == 'train':  return 50000
        elif self.split == 'val':  return 20000    

    def __getitem__(self, idx):
        idx = int(idx)
        subdir = str(idx - idx % 1000).zfill(5)
        path = f"{self.root}/thumbnails128x128/{subdir}/{str(idx).zfill(5)}.png"
        image = np.asarray(PIL.Image.open(path)) / 255.
        image = image * 2 - 1
        image = torch.tensor(image, dtype=torch.float).permute(2,0,1)
        return {'image': image}

    def set_growth_signal(self, growth_signal):
        self.growth_signal = growth_signal



class InfiniteSampler(Sampler):
    
    def __init__(self, dataset_size, shuffle=True):
        self.dataset_size = dataset_size
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            if self.shuffle: yield from torch.randperm(self.dataset_size)
            else:            yield from torch.arange(self.dataset_size)