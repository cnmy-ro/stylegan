import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import PIL



class FFHQ128x128Dataset(Dataset):

    def __init__(self, root, split):
        super().__init__()
        self.root = root
        self.split = split

    
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



class InfiniteSampler(Sampler):
    
    def __init__(self, size, shuffle=True):
        self.size = size
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size)
            else:
                yield from torch.arange(self.size)