import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd.variable import Variable
from torchvision import datasets, models, transforms

class Dog_Net(nn.Module):
    def __init__(self):
        super(Dog_Net, self).__init__()
        #load the trained inception network
        model = models.inception_v3(pretrained=True)
        self.freeze_model_parameters(model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1024)
        model.eval()
        self.inception = model

        # #first fully connected linear layer
        # self.fc1 = nn.Linear(2048, 1024)
        # #second fully connected linear layer

        self.fc2 = nn.Linear(1024, 120)
        #softmax
        # self.softmax = nn.Softmax(120)

    def forward(self, x):
        x = self.inception(x)
        # x = self.fc1(x)
        x = self.fc2(x)
        x = nn.Softmax(x)
        return x

    def freeze_model_parameters(self, model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False


if __name__ == "__main__":
    model = Dog_Net()
    child_counter = 0
    for child in model.children():
        print("   child", child_counter, "is -")
        print(child)
        child_counter += 1
