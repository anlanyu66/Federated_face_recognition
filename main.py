# single model face recognition

# dataset: LFW

import os
import time
import torch
import torch.optim as optim
import itertools
import torchvision
import numpy as np
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from Alexnet import Alexnet
import torchvision.models as models

if __name__ == '__main__':

    train_dir = 'train'
    dev_dir = 'dev'

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))
                                    ])
    embedding_dim = 64
    # model_name = 'lenet'
    # model_name = 'resnet152'
    model_name = 'alexnet'
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    if model_name == 'lenet':
        cnn = LeNet()
    elif model_name == 'alexnet':
        cnn = Alexnet(embedding_dim=embedding_dim, pretrained=False)
    elif model_name == 'resnet152':
        cnn = Resnet152(embedding_dim=embedding_dim, pretrained=False)

    num_epochs = 100
    minibatch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-4

    train_dataset = datasets.ImageFolder(train_dir, transform)
    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=minibatch_size,
                             shuffle=True, num_workers=0)

    dev_dataset = datasets.ImageFolder(dev_dir, transform)
    devLoader = torch.utils.data.DataLoader(dev_dataset, batch_size=8,
                                              shuffle=True, num_workers=0)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    gpu_device = 1

    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        total_train = 0
        correct_train = 0
        accuracy_train = 0
        for data in trainLoader:
            image, label = data
            cnn.train()
            optimizer.zero_grad()
            output = cnn.forward(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == label.view(*top_class.shape)
            accuracy_train += torch.mean(equals.type(torch.FloatTensor)).item()

        cnn.eval()
        total = 0
        correct = 0
        accuracy_val = 0
        with torch.no_grad():
            for val_data in devLoader:
                image_val, label_val = val_data
                output_val = cnn.forward(image_val)
                loss = criterion(output_val, label_val)
                val_loss += loss.item()

                ps = torch.exp(output_val)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == label_val.view(*top_class.shape)
                accuracy_val += torch.mean(equals.type(torch.FloatTensor)).item()

        loss_train.append(train_loss / len(trainLoader))
        loss_val.append(val_loss / len(devLoader))
        acc_train.append(100 * accuracy_train / len(trainLoader))
        acc_val.append(100 * accuracy_val / len(devLoader))
        print('epoch: {}, train loss: {}, val loss: {}, train acc: {}, val acc: {}'.format(epoch,
                                                                                           train_loss / len(
                                                                                               trainLoader),
                                                                                           val_loss / len(devLoader),
                                                                                           100 * accuracy_train / len(
                                                                                               trainLoader),
                                                                                           100 * accuracy_val / len(
                                                                                               devLoader)))

        cnn.train()
    np.save('testloss_1_worker', loss_val)
    np.save('trainloss_1_worker', loss_train)
    # np.save('testacc_1_worker', acc_val)
    # np.save('trainacc_1_worker', acc_train)

