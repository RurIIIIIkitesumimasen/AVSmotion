import os
import random
import pickle
import numpy as np
from tqdm import tqdm

import wandb
from wandb import log_artifact

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations

def transuse(a):
  i,  h, w = a.shape
  x = np.zeros((3, i, h, w))
  x[0] = a
  x[1] = a
  x[2] = a
  x = np.transpose(x, (1, 0, 2, 3))
  #print(x.shape)
  #x = x.squeeze(1).permute(0, 3, 1, 2)
  return x

class SetData():
    def __init__(self, object_array, root, img_size, seed, noise, noise_num):
        self.train_data, self.train_label, self.valid_data, self.valid_label = self.import_dataset(
            object_array, root, img_size, seed, noise, noise_num)

    def import_dataset(self, object_array, root, img_size, seed, noise, noise_num):

        train_data = []
        train_label = []
        test_data = []
        test_label = []

        train_i = 0
        test_i = 0

        if noise == True:
            root = root + "/" + noise_num

        for load_object in object_array:
            # print(os.getcwd())
            # load dataset
            x = np.load(#'orientation-detection/dataset/image/over32vbpixel-gs.npy')
            "MDD/dataset/Object_" + #"1pixel_32x32_data.pkl.npy"#small_size_image/gsimage/" +#"dataset/image/over32wbDATA.npy" )
                        str(load_object) + "pixel_32x32_data.pkl.npy"
                        )
            x1 = np.load(#'orientation-detection/dataset/image/over32vbpixel-gs.npy')
            "MDD/dataset/Object_" + #"1pixel_32x32_data.pkl.npy"#small_size_image/gsimage/gauss0，1/" +#"dataset/image/over32wbDATA.npy" )
                        str(load_object) + "pixel_32x32_data.pkl.npy"
                        )#noised
            t = np.load(#'orientation-detection/dataset/image/img_label/over32vbpixel-gs.npy')
            "MDD/dataset/Object_" +#small_size_image/label/" +#"dataset/image/over32wbLABEL.npy" )
                        str(load_object) + "pixel_32x32_label.pkl.npy"
                        )

            perm = np.random.RandomState(seed=seed).permutation(len(x))
            x = x[perm]
            x1 = x1[perm]
            t = t[perm]
               

            # Training Test ratio75：25
            train_data[train_i:train_i+7500] = x[:7500] * 1
            train_label[train_i:train_i+7500] = t[:7500] * 1
            train_i += 7500

            test_data[test_i:test_i+2500] = x1[7500:10000] * 1
            test_label[test_i:test_i+2500] = t[7500:10000] * 1
            test_i += 2500

            #print("import " +str(load_object) + "pixel dataset" + "rate" + train_i + ":" + test_i )

        train_data = np.array(train_data)
        train_label = np.array(train_label)
        train_label = np.argmax(train_label, axis=1)

        test_data = np.array(test_data)
        test_label = np.array(test_label)
        test_label = np.argmax(test_label, axis=1)

        perm_train = np.random.RandomState(
            seed=seed).permutation(len(train_data))
        train_data = train_data[perm_train][:3000]
        train_label = train_label[perm_train][:3000]
        perm_test = np.random.RandomState(
            seed=seed).permutation(len(test_data))
        test_data = test_data[perm_test][:500]
        test_label = test_label[perm_test][:500]

        return train_data, train_label, test_data, test_label

    def set_train_data_Loader(self, batch_size):
        train_dataset = MotionDetectionDataset(
            data=self.train_data, label=self.train_label)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )

        return train_loader

    def set_valid_data_Loader(self, batch_size):
        valid_dataset = MotionDetectionDataset(
            data=self.valid_data, label=self.valid_label)

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )

        return valid_loader


class MotionDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data)
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx].to(torch.float32)
        out_label = self.label[idx]

        # channel
        # return out_data[np.newaxis, :, :], out_label
        return out_data, out_label #EfN用
