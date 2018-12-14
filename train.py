import torch
import torch.optim as optim
from net import Dog_Net
from datasets import DogDataset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


def pad_collate(batch):
    """
    args:
        batch - list of (tensor, label)

    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    # find longest sequence
    max_len = max(map(lambda x: x[0].shape[0], batch))
    # pad according to max_len
    batch = map(lambda x, y:
                (pad_tensor(x, pad=max_len, dim=0), y), batch)
    # stack all
    x_tensors = []
    y_labels = []
    for item, y in list(batch):
        x_tensors.append(item)
        y_labels.append(y)
    xs = torch.stack(x_tensors, dim=0)
    ys = torch.LongTensor(y_labels)
    return xs, ys


def freeze_model_parameters(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def train(epochs=1000, mbsize=3, lr=0.0001):

    net = Dog_Net()
    # net.eval()
    train_data = DogDataset('train_list.mat', 'data')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    #load inception
    # inception = models.inception_v3(pretrained=True)
    # freeze_model_parameters(inception)
    # num_ftrs = inception.fc.in_features
    # inception.fc = nn.Linear(num_ftrs, 1024)
    # inception = nn.Sequential(*list(inception.children())[:-1])
    # new_children = list(inception.children())[:13] + list(inception.children())[13+1:]
    # inception = nn.Sequential(*new_children)

    # inception.aux_logit=False
    # inception.eval()
    # print(inception.training)
    # print(inception.aux_logit)
    # child_counter = 0
    # for child in inception.children():
    #     print("   child", child_counter, "is -")
    #     print(child)
    #     child_counter += 1
    trainloader = DataLoader(train_data, batch_size=mbsize,
                             shuffle=True, num_workers=4)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs = data['image']
            labels = data['label']
            # inputs, labels = data
            # print(inputs)
            #zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            # outputs = inception(inputs)
            # print("finished running inception")
            # print(outputs.shape)
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
