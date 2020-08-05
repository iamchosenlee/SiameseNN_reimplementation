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
    """select idxs from 20 alphabets for valid and test"""
    idx_val = random.sample(range(20), 10)
    idx_test = list(set(range(20)) - set(idx_val))

    return {'valid' : idx_val,
            'test' : idx_test}

def create_data_dict(root_dir, idx_alp=[], idx_drawers=[]):
    data_dict = defaultdict(dict) #data_dict[alphabet][character] = [list of images from different drawers]
    for idx, alp in enumerate(sorted(os.listdir(root_dir))):
        if (idx_alp==[]) or (idx in idx_alp):
            for char in sorted(os.listdir(root_dir + alp)):
                if idx_drawers==[]:
                    data_dict[alp][char] = os.listdir(os.path.join(root_dir, alp, char))
                else:
                    data_dict[alp][char] = [os.listdir(os.path.join(root_dir, alp, char))[i] for i in idx_drawers]
        else:
            pass
    return data_dict



class OmniglotDatasetDataset(Dataset):
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
        img1, img2, label = self.datas[idx]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

    def uniform_alp_count(self, alp_cnt, alps, count=1):
        for a in alps:
            alp_cnt[a] -= count
            if alp_cnt[a]<=0:
                del self.data_dict[a]
        return

    def get_pairs(self):
        paired_data = []
        alp_max = (self.size/len(self.data_dict.keys()))*2
        alp_cnt = {alp: alp_max for alp in list(self.data_dict.keys())}
        print("creating paired dataset...")
        for idx in range(self.size):
            if idx%2 == 0: #same character
                alp1, char1, drawer1, img_dir1 = self.sample_node()
                img_dir2 = self.sample_node(alp1, char1)[-1]
                self.uniform_alp_count(alp_cnt, [alp1], 2)
                label = 0.0
            else: # different character
                alp1, char1, drawer1, img_dir1 = self.sample_node()
                alp2, char2, drawer2, img_dir2 = self.sample_node()
                while (alp1, char1) == (alp2, char2):
                    img_dir2 = self.sample_node(alp2)[-1]
                self.uniform_alp_count(alp_cnt, [alp1, alp2], 1)
                label = 1.0
            img1 = Image.open(img_dir1).convert('L')
            img2 = Image.open(img_dir2).convert('L')
            paired_data.append((img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))))
        print("{} pairs created.".format(self.size))
        return paired_data


    def sample_node(self, alp=None, char=None):
        if not alp:
            alp = random.choice(list(self.data_dict.keys()))
        if not char:
            char = random.choice(list(self.data_dict[alp].keys()))        
        drawer = random.choice(list(self.data_dict[alp][char]))
        img_dir = os.path.join(self.root_dir, alp, char, drawer)

        return alp, char, drawer, img_dir



class NWayOneShotEvalSet(Dataset, OmniglotDatasetDataset):
    '''
        categories is the list of different alphabets (folders)
        root_dir is the root directory leading to the alphabet files, could be /images_background or /images_evaluation
        setSize is the size of the train set and the validation set combined
        numWay is the number of images (classes) you want to test for evaluation
        transform is any image transformations
    '''
    def __init__(self, root_dir, alphabets, drawers, numWay=20, numTrials = 20, transform=None):
        super().__init__(root_dir, drawers, size, transform)
        self.data_dict = create_data_dict(root_dir, idx_alp=alphabets, idx_drawers=drawers)
        self.numWay = numWay
        self.numTrials = numTrials
        self.datas = get_trials()


    def __len__(self):
        return 2 * len(alphabets) * self.numWay


    def __getitem__(self, idx):
        test_img, waySet, label = self.datas[idx]
        if self.transform:
            test_img = self.transform(test_img)
            waySet = [self.transform(img) for img in waySet]
        return test_img, waySet, label
 

    def sample_node(self, drawer_idx, alp_idx=None, char=None):
        """이상하지만 drawer, alp는 idx로 fix"""
        if not alp_idx:
            alp = random.choice(list(self.data_dict.keys()))
        elif alp_idx:
            alp = list(self.data_dict.keys())[alp_idx]
        if not char:
            char = random.choice(list(self.data_dict[alp].keys()))      
        drawer = self.data_dict[alp][char][drawer_idx]
        img_dir = os.path.join(self.root_dir, alp, char, drawer)

        return alp, char, drawer, img_dir


    def get_trials(self):
        trials = []
        for alp_idx in alphabets:
            for dwr_1, dwr_2 in [(drawers[0], drawers[1]), (drawers[2], drawers[3])]:   #여기 수정
                # find one main image
                alp1, char1, drawer1, test_img_dir = super.sample_node(dwr_1, alp_idx)
                test_img = Image.open(test_img_dir).convert('L')
                # find n numbers of distinct images, 1 in the same set as the main
                waySet = []
                label = np.random.randint(self.numWay)
                for i in range(self.numWay):
                    if i == label:
                        alp1, char2, drawer2, way_img_dir = super.sample_node(dwr_2, alp_idx, char1)
                    else:
                        alp1, char2, drawer2, way_img_dir = super.sample_node(dwr_2)
                        while char1 == char2:
                            way_img_dir = super.sample_node(dwr_2, alp_idx)[-1]

                    way_img = Image.open(way_img_dir).convert('L')
                    waySet.append(way_img)
                trials.append((test_img, waySet, torch.from_numpy(np.array(np.array([label], dtype=np.float32)))))
        return trials

