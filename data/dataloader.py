import os
from glob import glob
from PIL import Image
from typing import Callable, Optional
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import torchvision.transforms as transforms

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper

def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    new_dataset = __DATASET__[name](root=root, **kwargs)
    return new_dataset

def get_all_imgs(dataloader):
    all_imgs = []
    for images in dataloader:
        all_imgs.append(images)
    all_images = torch.cat(all_imgs, dim=0)
    return all_images

def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader

def get_transform(name: str, **kwargs):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform

@register_dataset(name='nudity')
@register_dataset(name='inappropriate')
class NudityDataset(VisionDataset):
    def __init__(self, root: str, class_info:str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        root_path = os.path.join(root, class_info)
        # self.fpaths = sorted(glob(f'{root_path}/*.png', recursive=True))
        self.fpaths = sorted(glob(f'{root_path}/*.png', recursive=True) +
                     glob(f'{root_path}/*.jpg', recursive=True))
        
        # VRAM out of memory
        if len(self.fpaths) > 3200:
            self.fpaths = self.fpaths[:3200]
        
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    
@register_dataset(name='artists')
class NudityDataset(VisionDataset):
    def __init__(self, root: str, class_info:str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        root_path = os.path.join(root, class_info)
        self.fpaths = sorted(glob(f'{root_path}/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img