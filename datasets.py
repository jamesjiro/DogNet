import os
import torch
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io, transform
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class DogDataset(Dataset):

    def __init__(self, data_mat, data_dir):
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
        #mapping of dog breeds to numbers (0-119)
        self.label_mapping = {}
        #counter for adding new dog breeds
        self.counter = 0

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        item = self.data_set[idx][0][0]
        img_name = os.path.join(self.img_dir,item)
        image = io.imread(img_name)
        annotation_file = os.path.join(self.annot_dir, item.replace(".jpg", ""))
        source, label = self.parse_annotation(annotation_file)
        sample = {'image': image, 'label': label}
        image_tensor = torch.FloatTensor(image)
        return image_tensor, label

    def parse_annotation(self, annotation_file):
        source = ""
        label = ""
        f = open(annotation_file, 'r')
        for line in f:
            if "<database>" in line:
                for item in line.split("</database>"):
                    if "<database>" in item:
                        source = item[item.find("<database>") + len("<database"):]
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
        return source, label_number



if __name__ == "__main__":
    train_set = DogDataset('train_list.mat', 'data')
    fig = plt.figure()
    for i in range(len(train_set)):
        sample = train_set[i]
        if sample['image'].shape[2] != 3:
            print("NOT RGB BROH")
        print(i, sample['image'].shape, sample['label'], sample['source'])
        # print(sample['image'])
        # ax = plt.subplot(1, 4, i + 1)
        # plt.tight_layout()
        # ax.set_title('sample#{}, breed:{}, source:{}'.format(i, sample['label'], sample['source']))
        # ax.axis('off')
        #
        # if i == 3:
        #     plt.show()
        #     break
