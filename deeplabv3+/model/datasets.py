import os
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from glob import glob


class DeeplabDataset(Dataset):
    def __init__(self, data_root, istrain, transform, preprocessing, img_suffix):
        super(DeeplabDataset, self).__init__()
        self.transform = transform
        self.preprocessing = preprocessing
        self.data_root = data_root
        self.istrain = istrain
        self.img_suffix = img_suffix
        self.img_paths = sorted(glob(os.path.join(self.data_root, self.istrain, 'image') + f'/*.{self.img_suffix}'))
        self.label_paths = sorted(glob(os.path.join(self.data_root, self.istrain, 'label') + f'/*.{self.img_suffix}'))
        # print(self.label_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imdecode(np.fromfile(self.img_paths[idx], dtype=np.uint8), 1)
        label = cv2.imdecode(np.fromfile(self.label_paths[idx], dtype=np.uint8), 0)
        label = label[:, :, np.newaxis].astype('float')
        # print(label)
        # label[label >= 1] = 1

        # apply augmentations
        if self.transform:
            sample = self.transform(image=img, mask=label)
            img, label = sample['image'], sample['mask']
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=label)
            img, label = sample['image'], sample['mask']

        return img, label


class TestDataset(Dataset):
    def __init__(self, data_root, transform,preprocessing, img_suffix):
        super(TestDataset, self).__init__()
        self.data_root = data_root
        self.img_suffix = img_suffix
        self.transform = transform
        self.preprocessing = preprocessing
        self.img_paths = sorted(glob(self.data_root + f'/*.{self.img_suffix}'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = os.path.basename(self.img_paths[idx])
        img = cv2.imdecode(np.fromfile(self.img_paths[idx], dtype=np.uint8), 1)

        # apply augmentations
        if self.transform:
            sample = self.transform(image=img)
            img = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img)
            img = sample['image']
        return img, img_name


