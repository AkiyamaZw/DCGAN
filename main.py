import torch
import torch.nn as nn 
import torch.optim as optim 
import torchvision.utils as vutils
import numpy as np 
import random 

import dataset 
import param as p 
import net 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
def main():
    # Set random seem for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1,10000)
    print("Random Seed: ",manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # get device
    device = torch.device("cuda:1" if (torch.cuda.is_available() and p.ngpu>0) else "cpu")
    # get network
    netG = net.Generator(p.ngpu).to(device)
    netD = net.Discriminator(p.ngpu).to(device)
    if (device.type=='cuda' and (p.ngpu > 1)):
        netG = nn.DataParallel(netG,list(range(p.ngpu)))
        netD = nn.DataParallel(netD,list(range(p.ngpu)))
    #netG.apply(net.weights_init)
    #netD.apply(net.weights_init)
    print(netG)
    print(netD)

    # Loss function and optimizer
    criterion = nn.BCELoss()

    # create batch of latent vectors what we will use to visualize the progression of the generator
    fixed_noise = torch.randn(p.batch_size,p.nz,1,1,device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # setup Adam optimiziers for both G and D
    optimizerG = optim.Adam(netG.parameters(),lr=p.g_lr,betas=(p.beta1,0.999))
    optimizerD = optim.Adam(netD.parameters(),lr=p.d_lr,betas=(p.beta1,0.999))

    # start to train
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    for epoch in range(p.num_epochs):
        for i,data in enumerate(dataset.dataloader,0):
            # (1) Update D network: maximize log(D(x)) + log(1-D(G(z)))
            netD.zero_grad()
            # Format batch
            real = data.to(device)
            label = torch.full((data.size(0),),real_label,device=device)
            # Forward pass real batch through D
            output = netD(real).view(-1) # resize [batch_size,1,1,1] to [batch_size]
            errD_real = criterion(output,label)
            errD_real.backward()
            D_x = output.mean().item()

            #Generate batch of latent vectors
            noise = torch.randn(data.size(0),p.nz,1,1,device=device)
            #Generate batch of fake image
            fake = netG(noise)
            label.fill_(fake_label)
            #Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output,label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake 
            #errD.backward()
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output,label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()                        
            optimizerG.step()

            # output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.8f\tLoss_G: %.8f\tD(x): %.8f\tD(G(z)): %.8f / %.8f'
                  % (epoch, p.num_epochs, i, len(dataset.dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch==p.num_epochs-1) and (i==len(dataset.dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake,padding=2,normalize=True).numpy())
            iters += 1
    np.save(p.result_dir+"img_list.npy",img_list)
    after_train(D_losses,G_losses)
    show_after_train(img_list)

def after_train(D_losses,G_losses):
    fig1 = plt.figure(figsize=(10,5))
    ax = fig1.add_subplot(111)
    ax.set_title("Generator and Discriminator Loss During Training")
    ax.plot(D_losses,label="D")
    ax.plot(G_losses,label="G")
    ax.set_xlabel("iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(p.result_dir+"dgloss.jpg")
    plt.show()

def show_after_train(img_list):
    if path:
    fig = plt.figure(figsize=(8,8))
    plt.axis('off')
    ims = [[plt.imshow(np.transpose(i,(1,2,0)),animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig,ims,interval=1000,repeat_delay=1000,blit=True)
    ani.save(p.result_dir+"anima.gif",writer='imagemagick',fps=3)
    plt.show()

if __name__ =="__main__":
    main()