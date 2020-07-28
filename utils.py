import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict
import random


ROOT_DIR = './dataset/images_background/'
random_seed = 0
np.random.seed(random_seed)


class OmniglotDataset(Dataset):
    def __init__(self, root_dir=ROOT_DIR, size = 19280, transform=None):
        self.root_dir = root_dir
        self.size = size
        self.data_dict = defaultdict(dict) #data_dict[alphabet][character] = [list of images]
        for alp in os.listdir(root_dir):
            for char in os.listdir(root_dir + alp):
                self.data_dict[alp][char] = os.listdir(os.path.join(root_dir, alp, char))
        self.transform = transform
        
    def __len__(self):
        return self.size

    def __getitem__(self,idx):
        img1 = None
        img2 = None 
        label = None
        if idx%2 == 0: #same character
            alp = random.choice(list(data_dict.keys()))
            char = random.choice(list(data_dict[alp].keys()))
            img_dir = os.path.join(self.root_dir, alp, char)
            images = random.sample(os.listdir(img_dir), 2)
            img1, img2 = Image.open(os.path.join(img_dir,images[0])), Image.open(os.path.join(img_dir,images[1]))
            label = 0.0
        else: # different character
            alp1 , alp2 = random.choice(list(data_dict.keys())), random.choice(list(data_dict.keys()))
            char1, char2 = random.choice(list(data_dict[alp1].keys())), random.choice(list(data_dict[alp2].keys()))
            img_dir1, img_dir2 = os.path.join(self.root_dir, alp1, char1), os.path.join(self.root_dir, alp2, char2)
            images = random.choice(data_dict[alp1][char1]), random.choice(data_dict[alp2][char2])
            while (img_dir1, images[0]) == (img_dir2, images[1]):
                images[1] = random.choice(data_dict[alp2][char2])
            img1, img2 = Image.open(img_dir1+ '/' + images[0]), Image.open(img_dir2 + '/' + images[1])
            label = 1.0
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))