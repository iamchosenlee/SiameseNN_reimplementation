import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from collections import defaultdict

ROOT_DIR = './dataset/images_background/'
random_seed = 0
np.random.seed(random_seed)

def split_drawers():
    """select idxs from 20 drawers for each occasion
        idx_background : 12
        idx_eval : 8
            idx_valid : 4
            idx_test : 4
    """
    idx_eval = np.random.choice(range(20), 8, replace=None)
    idx_test = np.random.choice(idx_eval, 4, replace=None)
    idx_valid = list(set(idx_eval) - set(idx_test))
    idx_background = list(set(range(20)) - set(idx_eval))

    return {'eval' : idx_eval, 
            'test' : idx_test,
            'valid' : idx_valid}

def split_alphabets():
    """select idxs from 20 alphabets for valid(10) and test(10)"""
    idx_val = random.sample(range(20), 10)
    idx_test = list(set(range(20)) - set(idx_val))

    return {'valid' : idx_val,
            'test' : idx_test}

def create_data_dict(root_dir, idx_alps=[], idx_drawers=[]):
    data_dict = defaultdict(dict) #data_dict[alphabet][character] = [list of images from different drawers]
    for idx, alp in enumerate(sorted(os.listdir(root_dir))):
        if (idx_alps==[]) or (idx in idx_alps):
            for char in sorted(os.listdir(root_dir + alp)):
                if idx_drawers==[]:
                    data_dict[alp][char] = os.listdir(os.path.join(root_dir, alp, char))
                else:
                    data_dict[alp][char] = [os.listdir(os.path.join(root_dir, alp, char))[i] for i in idx_drawers]
        else:
            pass
    return data_dict



class OmniglotDataset(Dataset):
    """verification datset"""
    def __init__(self, root_dir, drawers, size = 90000, transform=None):
        self.root_dir = root_dir
        self.size = size
        self.data_dict = create_data_dict(root_dir, idx_drawers=drawers)
        self.transform = transform
        self.datas = self.get_pairs()
        
    def __len__(self):
        return self.size

    def __getitem__(self,idx):
        # img_dir1, img_dir2, label = self.datas[idx]
        # img1 = Image.open(img_dir1).convert('L')
        # img2 = Image.open(img_dir2).convert('L')
        img1, img2, label = self.datas[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def uniform_alp_count(self, alp_cnt, alps, count=1):
        """if an alphabet is sampled more than alp_cnt, delete the alphabet from data_dict"""
        try:
            for a in alps:
                alp_cnt[a] -= count
                if alp_cnt[a]<=0:
                    #print(a, alp_cnt[a])
                    self.data_dict.pop(a)
        except KeyError:
            pass
        return

    def get_pairs(self):
        """creates paired data with label"""
        paired_data = []
        alp_max = (self.size/len(self.data_dict.keys()))*2
        alp_cnt = {alp: alp_max for alp in list(self.data_dict.keys())}
        print("creating paired dataset...")
        for idx in range(self.size):
            if idx%2 == 0: #same character
                alp1, char1, drawer1, img_dir1 = self.sample_img()
                img_dir2 = self.sample_img(alp1, char1)[-1]
                self.uniform_alp_count(alp_cnt, [alp1], 2)
                label = 0.0
            else: # different character
                alp1, char1, drawer1, img_dir1 = self.sample_img()
                alp2, char2, drawer2, img_dir2 = self.sample_img()
                while (alp1, char1) == (alp2, char2):
                    img_dir2 = self.sample_img(alp2)[-1]
                self.uniform_alp_count(alp_cnt, [alp1, alp2], 1)
                label = 1.0
            img1 = Image.open(img_dir1).convert('L')
            img2 = Image.open(img_dir2).convert('L')
            paired_data.append((img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))))
            #paired_data.append((img_dir1, img_dir2, label))
        print("{} pairs created.".format(self.size))
        return paired_data


    def sample_img(self, alp=None, char=None):
        """ random sample one img
            can fixate alp or char when needed
            returns chosen alp, char, drawer, and full img_dir"""
        if not alp:
            alp = random.choice(list(self.data_dict.keys()))
        if not char:
            char = random.choice(list(self.data_dict[alp].keys()))        
        drawer = random.choice(list(self.data_dict[alp][char]))
        img_dir = os.path.join(self.root_dir, alp, char, drawer)

        return alp, char, drawer, img_dir



class NWayOneShotEvalSet(Dataset):
    def __init__(self, root_dir, idx_alps, idx_drawers, numWay=20, numTrials = 20, transform=None):
        self.root_dir = root_dir #'/images_background' or '/images_evaluation'
        self.idx_alps = idx_alps
        self.idx_drawers = idx_drawers
        self.data_dict = create_data_dict(root_dir, idx_alps=self.idx_alps, idx_drawers=self.idx_drawers)
        self.numWay = numWay
        self.numTrials = numTrials
        self.transform = transform
        self.datas = self.get_trials()


    def __len__(self):
        return 2 * len(self.idx_alps) * self.numWay #400


    def __getitem__(self, idx):
        test_img, waySet, label = self.datas[idx]
        if self.transform:
            test_img = self.transform(test_img)
            waySet = [self.transform(img) for img in waySet]
        return test_img, waySet, label
 

    def get_dir(self, alp_idx, char_idx, drawer_idx):
        """get img_dir from alp, char, drawer's idx"""
        alp = list(self.data_dict.keys())[alp_idx]
        char = list(self.data_dict[alp].keys())[char_idx]
        return os.path.join(self.root_dir, alp, char, self.data_dict[alp][char][drawer_idx])


    def get_trials(self):
        trials = []
        print("creating {} way one shot evaluation dataset...".format(self.numWay))

        # for 10 alphabets
        for alp_idx in range(len(self.idx_alps)):
            alp = list(self.data_dict.keys())[alp_idx]
            # choose 20 random characters_idx for 20 ways
            idx_chars = random.sample(range(len(self.data_dict[alp].keys())), self.numWay)
            # for 2 pairs [(dwr1, dwr2), (dwr3, dwr4)]
            for dwr_idx1, dwr_idx2 in [(0,1), (2,3)]:
                # find one character for main image
                label = np.random.randint(self.numWay)
                test_img_dir = self.get_dir(alp_idx, idx_chars[label], dwr_idx1)
                test_img = Image.open(test_img_dir).convert('L')
                # find n numbers of distinct images, 1 in the same set as the main
                waySet = []

                for i in range(self.numWay):
                    way_img_dir = self.get_dir(alp_idx, idx_chars[i], dwr_idx2)
                    way_img = Image.open(way_img_dir).convert('L')
                    waySet.append(way_img)
                trials.append((test_img, waySet, torch.from_numpy(np.array([label], dtype=np.float32))))
        print("{} trials created".format(self.numTrials))
        return trials

