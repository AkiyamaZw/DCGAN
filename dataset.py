import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.utils as vutils
import torchvision.datasets as dset 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import param as p 
import numpy as np 
import os 
from PIL import Image 

class CelebA(Dataset):
    def __init__(self,rootdir,transform=None):
        self.rootdir = rootdir
        self.transform=transform
        self.filenames = os.listdir(self.rootdir)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self,idx):
        path = os.path.join(self.rootdir,self.filenames[idx])
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image 

transform=transforms.Compose([transforms.Resize(p.image_size),
                              transforms.CenterCrop(p.image_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
dataset = CelebA(p.dataroot,transform=transform)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=p.batch_size,shuffle=True,num_workers=p.workers)

if __name__ =="__main__":
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title("celeba data")
    imgs = vutils.make_grid(real_batch[:64],padding=2,normalize=True).numpy()
    print(type(imgs))
    plt.imshow(np.transpose(imgs,(1,2,0)))
    plt.show()