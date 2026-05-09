from __future__ import print_function
from torch.utils.data import Dataset
from torchvision import transforms 

##############################################################################
class satellitedata(Dataset):
    def __init__(self,images,masks, transform=None):
        
        self.images = images
        self.masks = masks
        self.transform = transform
        
    def __getitem__(self, index):
        
        im = self.images[index]
        ma = self.masks[index]
        
        if self.transform:
            im = self.transform(im)
            ma = self.transform(ma)

        return im, ma
        
    def __len__(self):

        return len(self.images)

##############################################################################