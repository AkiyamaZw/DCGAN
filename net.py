import torch
import torch.nn as nn 
import torch.nn.functional as F 
import param as p 
# for debug
import matplotlib.pyplot as plt  
import numpy as np 

class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu 
        self.main = nn.Sequential(nn.ConvTranspose2d(p.nz,p.ngf*8,4,1,0,bias=False),
                                  nn.BatchNorm2d(p.ngf*8),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(p.ngf*8,p.ngf*4,4,2,1,bias=False),
                                  nn.BatchNorm2d(p.ngf*4),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(p.ngf*4,p.ngf*2,4,2,1,bias=False),
                                  nn.BatchNorm2d(2*p.ngf),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(p.ngf*2,p.ngf,4,2,1,bias=False),
                                  nn.BatchNorm2d(p.ngf),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(p.ngf,p.nc,4,2,1,bias=False),
                                  nn.Tanh())

    def forward(self,inputs):
        return self.main(inputs)

class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator,self).__init__()
        self.ngpu=ngpu 
        self.main = nn.Sequential(nn.Conv2d(p.nc,p.ndf,4,2,1,bias=False),
                                  nn.BatchNorm2d(p.ndf),
                                  nn.LeakyReLU(0.2,True),
                                  nn.Conv2d(p.ndf,p.ndf*2,4,2,1,bias=False),
                                  nn.BatchNorm2d(p.ndf*2),
                                  nn.LeakyReLU(0.2,True),
                                  nn.Conv2d(p.ndf*2,p.ndf*4,4,2,1,bias=False),
                                  nn.BatchNorm2d(p.ndf*4),
                                  nn.LeakyReLU(0.2,True),
                                  nn.Conv2d(p.ndf*4,p.ndf*8,4,2,1,bias=False),
                                  nn.BatchNorm2d(p.ndf*8),
                                  nn.LeakyReLU(0.2,True),
                                  nn.Conv2d(p.ndf*8,1,4,1,0,bias=False),
                                  nn.Sigmoid())
    def forward(self,inputs):
        return self.main(inputs)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find("BatchNorm")!=-1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.normal_(m.bias.data,0)

if __name__ == "__main__":
    a = torch.rand(1,100,1,1)
    print(a.requires_grad)
    g = Generator(1)
    output = g(a)
    print(output.size())
    print(type(output))
    print(output.requires_grad)
    out = output.detach()
    print(out.requires_grad)
    print(type(out))
    img = np.transpose(out.numpy()[0],(1,2,0))
    plt.imshow(img)
    plt.show()
    d = Discriminator(1)
    out = d(output)
    print(type(out))
    print(out.detach().numpy())
    print(out.item())
    x = out.view(-1)
    print(type(x))
    print(out.size())
    print(x.size())
