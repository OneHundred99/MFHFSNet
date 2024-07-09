from os.path import exists, join
from torchvision.transforms import Compose, CenterCrop, ToTensor
import torch.utils.data as data
from os import listdir
from PIL import Image
import torch
import os


destination = "/mnt/b/MICCAI_BraTS_2019_Data_Training/data/"


class DataGet(torch.utils.data.Dataset):
    def __init__(self, img_dir, anno_dir, transform=None):
        img_ilist = []
        img_alist = []
        self.img_dir = join(destination, img_dir)
        self.anno_dir = join(destination, anno_dir)
        for file in os.listdir(self.img_dir):
            img_ilist.append(os.path.join(img_dir, file))
            img_alist.append(os.path.join(anno_dir, file))
        self.img_ilist = img_ilist
        self.img_alist = img_alist
        self.transform = transform

    def __getitem__(self, index):
        img_ipath = join(destination, self.img_ilist[index])
        image = Image.open(img_ipath).convert('L')
        image = image.resize((512, 512))

        img_apath = join(destination, self.img_alist[index])
        label = Image.open(img_apath)
        label = label.resize((512, 512))

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_ilist)
