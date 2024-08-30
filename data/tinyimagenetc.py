import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torchvision
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_dataset_tinyimagenet_c(path, ctype, cdeg, label_mapper):
    count_dict = {}
    count_dict["val"] = {}

    val_images = []
    val_labels = []

    
    for cl in os.listdir(os.path.join(path, ctype, cdeg)):
        for file in os.listdir(cl):
            val_images.append(mpimg.imread(file))
            lab = label_mapper[cl]
            val_labels.append(lab)
            if lab in count_dict.keys(): count_dict[lab] += 1
            else: count_dict[lab] = 1
    dataset = {}
    dataset["val"] = []
    for i in range(len(val_images)):
        dataset["val"].append({})
        dataset["val"][i]["image"] = val_images[i]
        dataset["val"][i]["label"] = val_labels[i]

    return dataset, None, count_dict


