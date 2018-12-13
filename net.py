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
        self.inception = models.inception_v3(pretrained=True)
        #cut off the final linear layer making the output a 2048 vector
        self.inception = nn.Sequential(*list(self.inception.children())[:-1])
        #freeze all parameters in the model
        self.freeze_model_parameters(self.inception)
        #first fully connected linear layer
        self.fc1 = nn.Linear(2048, 1024)
        #second fully connected linear layer
        self.fc2 = nn.Linear(1024, 120)
        #softmax
        self.softmax = nn.Softmax(120)

    def forward(self, x):
        x = self.inception(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
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
