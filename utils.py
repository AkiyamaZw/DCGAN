import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import param as p

def show_from_npy(path):
    img_list = np.load(path)
    fig = plt.figure(figsize=(8,8))
    plt.axis('off')
    ims = [[plt.imshow(np.transpose(i,(1,2,0)),animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig,ims,interval=1000,repeat_delay=1000,blit=False)
    plt.margins(0,0)
    ani.save(p.result_dir+"anima.gif",writer='imagemagick',fps=10)
    plt.show()

if __name__ == "__main__":
    show_from_npy("./results/img_list.npy")