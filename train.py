import os
import torch
from torch.utils.data import DataLoader
from utils import OmniglotDataset
import matplotlib.pyplot as plt
from torchvision import transforms
random_seed = 0
torch.manual_seed(random_seed)

ROOT_DIR = './dataset/images_background/'


if __name__ == '__main__':
    transformations = transforms.Compose([transforms.ToTensor()]) 
    trainset = OmniglotDataset(root_dir = ROOT_DIR, size=100, transform=transformations)
    loader = DataLoader(trainset, batch_size = 1)
    
    for idx, (img1, img2, label) in enumerate(loader):
        if idx <2:
            print(label[0])
            plt.subplot(2,2,1)
            plt.imshow(img1[0][0])
            plt.subplot(2,2,2)
            plt.imshow(img2[0][0])
            plt.show()
        else:
            break
