import torch
import time
import copy
import json
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from net import Dog_Net
from datasets import DogDataset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
import itertools

def test(label_mapping, mbsize=32):
    model = Dog_Net()
    model.load_state_dict(torch.load('checkpoint39.path.tar'))
    model.eval()
    model.cuda()

    test_data = DogDataset('test_list.mat', 'data', False, label_mapping)
    test_loader = DataLoader(test_data, batch_size=mbsize, shuffle=True,
                             num_workers=4, pin_memory=True)

    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images = data['image']
            labels = data['label']
            images, labels = images.cuda(), labels.cuda()
            ouputs = model(images)
            _, predicted = torch.max(ouputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true += labels.tolist()
            y_pred += predicted.tolist()
    print('F1_score with macro: {} '.format(
        metrics.f1_score(y_true, y_pred, average='micro')))
    print('Accuracy of the network on the images: %d %%' %(
        100 * correct / total))

    mat = confusion_matrix(y_true, y_pred)
    np.save('confusion_array.np', mat)
    return mat
    #print("-------matrix------\n", mat)


def labels_to_list(label_mapping, num_labels):
    label_list = ["" for i in range(num_labels)]
    for label in label_mapping:
        label_list[label_mapping[label]] = label
    print(label_list)
    return label_list



def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title='DogNet Confusion matrix',
                          cmap=plt.cm.tab20c):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
     #   plt.text(j, i, format(cm[i, j], fmt),
      #           horizontalalignment="center",
       #          color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.tight_layout()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, loader, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data['image']
            labels = data['label']
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    return
        model.train(mode=was_training)

def make_bar_chart(confusion_matrix, label_mapping):
    label_list = labels_to_list(label_mapping, 120)
    confusion_matrix = np.load('confusion_array.np.npy')
    rows = len(confusion_matrix)
    max_idxs = []
    for row in range(rows):
        for col in range(rows):
            if row == col:
                continue
            val = confusion_matrix[row, col]
            max_idxs.append((val, [row, col]))
    max_idxs = sorted(max_idxs, key=lambda x: x[0], reverse=True)
    top = max_idxs[:10]
    names = [i[1] for i in top]
    vals = [i[0] for i in top]
    for i in range(len(names)):
        names[i] = label_list[names[i][0]] + "/" + label_list[names[i][1]]
    xs = np.arange(len(vals))

    plt.bar(xs, vals, align='center', color="orange")
    plt.xticks(xs, names, rotation=90)
    plt.ylabel('Number of misclassifications')
    plt.title('Most commonly misclassified doge breeds')
    plt.show()

if __name__ == '__main__':
    with open('label_mapping.json') as json_mapping:
        label_mapping = json.load(json_mapping)
    label_list = labels_to_list(label_mapping, 120)
    confusion_matrix = np.load('confusion_array.np.npy')
    make_bar_chart(confusion_matrix, label_mapping)
    plt.figure()
    plot_confusion_matrix(confusion_matrix, label_list)
    plt.show()

    test_data = DogDataset('test_list.mat', 'data', False, label_mapping)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True,
                             num_workers=4, pin_memory=True)

    net = Dog_Net()
    net.load_state_dict(torch.load('checkpoint39.path.tar'))
    net.eval()
    net.cuda()
    visualize_model(net, test_loader, label_list, 10)
