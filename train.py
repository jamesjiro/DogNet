import torch
import time
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

    net = Dog_Net().cuda()
    # net.eval()
    train_data = DogDataset('train_list.mat', 'data')
    train_data.populate_labels()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    trainloader = DataLoader(train_data, batch_size=mbsize, shuffle=True,
                             num_workers=4, pin_memory=True)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # print("training on batch {}".format(i))
            inputs = data['image']
            labels = data['label']
            inputs = inputs.cuda()
            labels = labels.cuda()

            # compute output
            output = net(inputs)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
            losses.update(loss, inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # zero the gradient do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                          epoch, i, len(trainloader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, top5=top5))

    torch.save(net.state_dict(), "trained_net/checkpoint.path.tar")

if __name__ == '__main__':
    train()
