
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torchvision

from data.Dataset import transform

class celebABlond(Dataset):
    def __init__(self, celebA, transform=None):
        self.dataset = celebA
        self.transform = transform
    
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img = self.dataset[index][0]
        lab = self.dataset[index][1][9]
        if self.transform:
            img = self.transform(img)
        return img, lab

def get_dataset_celebA(path):
    count_dict = {}
    count_dict["train"] = {}
    count_dict["val"] = {}
    celebAt = torchvision.datasets.CelebA(root = path, download= True)
    celebAv = torchvision.datasets.CelebA(root = path, download= False, split="valid")
    celebAte = torchvision.datasets.CelebA(root = path, download= False, split="test")
    data_t = celebABlond(celebAt)
    data_v = celebABlond(celebAv)
    data_te = celebABlond(celebAte)
    loader_t = DataLoader(data_t, batch_size=1)
    loader_v = DataLoader(data_v, batch_size=1)
    loader_te = DataLoader(data_te, batch_size=1)
    training_images = []
    training_labels = []
    val_images = []
    val_labels = []
    for (train_img, label) in iter(loader_t):
        training_images.append(train_img)
        training_labels.append(label)
        if label in count_dict["train"].keys(): count_dict["train"][label] = 1
        else: count_dict["train"][label] += 1
    
    for (val_img, label) in iter(loader_v):
        val_images.append(val_img)
        val_labels.append(label)
        if label in count_dict["val"].keys(): count_dict["val"][label] = 1
        else: count_dict["val"][label] += 1

    for (test_img, label) in iter(loader_te):
        val_images.append(test_img)
        val_labels.append(label)
        count_dict["val"][label] += 1
    


    dataset = {}
    dataset["train"] = {}
    dataset["val"] = {}   
    for i in range(len(training_images)):
        dataset["train"].append({})
        dataset["train"][i]["image"] = training_images[i]
        dataset["train"][i]["label"] = training_labels[i]
    
    for i in range(len(val_images)):
        dataset["val"].append({})
        dataset["val"][i]["image"] = val_images[i]
        dataset["val"][i]["label"] = val_labels[i]

    return dataset, None, count_dict
