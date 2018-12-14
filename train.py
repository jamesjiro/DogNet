import torch
import torch.optim as optim
from net import Dog_Net
from datasets import DogDataset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

def freeze_model_parameters(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def train(epochs=1000, mbsize=64, lr=0.0001):

    net = Dog_Net().cuda()
    # net.eval()
    train_data = DogDataset('train_list.mat', 'data')
    train_data.populate_labels()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    trainloader = DataLoader(train_data, batch_size=mbsize, shuffle=False,
                             num_workers=4, pin_memory=True)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            print("training on batch {}".format(i))
            inputs = data['image']
            labels = data['label']
            inputs = inputs.cuda()
            labels = labels.cuda()
            print("labels:", labels)
            print("inputs", inputs.shape)
            #zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i%100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss /100))
                running_loss = 0.0


train()
