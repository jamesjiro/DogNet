import torch
import torch.optim as optim
from net import Dog_Net
from datasets import DogDataset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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


# no longer needed with preprocessing transform step in datasets.py
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

def train(epochs=1000, mbsize=64, lr=0.0001):

    net = Dog_Net()
    train_data = DogDataset('train_list.mat', 'data')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    trainloader = DataLoader(train_data, batch_size=mbsize,
                             shuffle=True, num_workers=4)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            #zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
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
