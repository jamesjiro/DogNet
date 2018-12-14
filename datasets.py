import os
import torch
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io, transform
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image



class DogDataset(Dataset):

    def __init__(self, data_mat, data_dir, train=True):
        # normalization transform for Inception v3 model
        # (see https://pytorch.org/docs/stable/torchvision/models.html)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        if train:
            # transform taken from imagenet example for training
            transform = transforms.Compose([
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # transform for testing
            transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize,
            ])
        # root directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # data directory
        self.data_dir = os.path.join(dir_path, data_dir)
        # image directory
        self.img_dir = os.path.join(self.data_dir, 'Images')
        # annotation directory
        self.annot_dir = os.path.join(self.data_dir, 'Annotation')
        # train/test/all list directory
        self.list_dir = os.path.join(self.data_dir, 'lists')
        # file dictionary
        mat = sio.loadmat(os.path.join(self.list_dir, data_mat))
        #file list
        self.data_set = mat['file_list']
        # transform for preprocessing
        self.transform = transform
        #mapping of dog breeds to numbers (0-119)
        self.label_mapping = {}
        #counter for adding new dog breeds
        self.counter = 0

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        item = self.data_set[idx][0][0]
        img_name = os.path.join(self.img_dir,item)
        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)
        annotation_file = os.path.join(self.annot_dir, item.replace(".jpg", ""))
        source, label, annotation_file = self.parse_annotation(annotation_file)
        sample = {'image': image, 'label': label, 'file': annotation_file}
        return sample

    def populate_labels(self):
        for subdir in os.listdir(self.annot_dir):
            subdir_path = os.path.join(self.annot_dir, subdir)
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                self.parse_annotation(file_path)

    def parse_annotation(self, annotation_file):
        f = open(annotation_file, 'r')
        for line in f:
            if "<database>" in line:
                for item in line.split("</database>"):
                    if "<database>" in item:
                        source = item[item.find("<database>") + len("<database>"):]
            if "<name>" in line:
                for item in line.split("</name>"):
                    if "<name>" in item:
                        label = item[item.find("<name>") + len("<name>"):]
                        if label in self.label_mapping:
                            label_number = self.label_mapping[label]
                        else:
                            label_number = self.counter
                            self.label_mapping[label] = self.counter
                            self.counter += 1
        return source, label_number, annotation_file



if __name__ == "__main__":
    train_set = DogDataset('train_list.mat', 'data')
    fig = plt.figure()
    for i in range(len(train_set)):
        sample = train_set[i]
        if sample['image'].shape[0] != 3:
            print("NOT RGB BROH")
            print(i, sample['image'].shape, sample['file'])
