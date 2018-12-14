import torch
import time
import copy
import json
import torch.optim as optim
from net import Dog_Net
from datasets import DogDataset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def freeze_model_parameters(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def train(epochs=5000, mbsize=64, lr=0.0001):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    net = Dog_Net()
    # net.eval()
    train_data = DogDataset('train_list.mat', 'data', True)
    train_data.populate_labels()
    with open('label_mapping.json', 'w') as outfile:
        json.dump(train_data.label_mapping, outfile)
    test_data = DogDataset('test_list.mat', 'data', False, label_mapping=train_data.label_mapping)
    # test_data.populate_labels()
    dataset_sizes = {'train': len(train_data), 'val': len(test_data)}
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    train_loader = DataLoader(train_data, batch_size=mbsize, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_data, batch_size=mbsize, shuffle=True,
                            num_workers=4, pin_memory=True)

    # Initialize best model weights and accuracy
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                loader = train_loader
            else:
                loader = val_loader
                net.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(loader):
                # measure data loading time
                data_time.update(time.time() - end)

                # print("training on batch {}".format(i))
                inputs = data['image']
                labels = data['label']
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # compute output
                with torch.set_grad_enabled(phase == 'train'):
                    output = net(inputs)
                    _, preds = torch.max(output, 1)
                    loss = criterion(output, labels)

                    # backward, optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
                losses.update(loss, inputs.size(0))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5, inputs.size(0))


                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # running_loss += loss.item()

                if i % 10 == 9:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              epoch, i, len(loader), batch_time=batch_time,
                              data_time=data_time, loss=losses, top1=top1, top5=top5))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())

        if epoch % 200 == 199:
            torch.save(best_model_wts, "trained_net/checkpoint{}.path.tar")

if __name__ == '__main__':
    train()
