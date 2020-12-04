# federated learning face recognition

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

    # dataset directory: train1, train2, train3: IID data, equally split the images for each identity
    #                    train1_diff, train2_diff, train3_diff: non-IID data, identities are equally split
    train_dir = 'train'
    train_dir1 = 'train1'
    train_dir2 = 'train2'
    train_dir3 = 'train3'
    dev_dir = 'dev'
    test_dir = 'test'

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))
                                    ])
    embedding_dim = 64
    # model_name = 'lenet'
    # model_name = 'resnet'
    model_name = 'alexnet'
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    if model_name == 'lenet':
        cnn = LeNet()
    elif model_name == 'alexnet':
        cnn = Alexnet(embedding_dim=embedding_dim, pretrained=False)
        cnn1 = Alexnet(embedding_dim=embedding_dim, pretrained=False)
        cnn2 = Alexnet(embedding_dim=embedding_dim, pretrained=False)
        cnn3 = Alexnet(embedding_dim=embedding_dim, pretrained=False)
    elif model_name == 'resnet152':
        cnn = Resnet152(embedding_dim=embedding_dim, pretrained=False)

    num_epochs = 100
    minibatch_size = 8
    learning_rate = 1e-4
    inner_epochs = 5
    num_worker = 3

    dev_dataset = datasets.ImageFolder(dev_dir, transform)
    devLoader = torch.utils.data.DataLoader(dev_dataset, batch_size=8,
                                            shuffle=True, num_workers=0)

    test_dataset = datasets.ImageFolder(test_dir, transform)
    testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=8,
                                             shuffle=True, num_workers=0)

    trainLoader = []
    for i in range(num_worker):
        train_dataset = datasets.ImageFolder('train{}_diff'.format(i+1), transform)
        trainLoader.append(torch.utils.data.DataLoader(train_dataset, batch_size=minibatch_size,
                                                       shuffle=True, num_workers=0))
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)
    optimizer3 = torch.optim.Adam(cnn3.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(cnn.parameters(), lr=learning_rate)

    acc_train = [[] for _ in range(num_worker)]
    acc_val = []
    acc_test = []
    weight1 = []
    weight2 = []
    weight3 = []

    loss_train = [[] for _ in range(num_worker)]
    loss_test = []

    param = cnn.state_dict()
    w = [param.copy(), param.copy(), param.copy()]

    for outer in range(num_epochs):
        total_train = 0
        correct_train = 0

        accuracy_val = np.zeros(3)
        for epoch in range(inner_epochs):
            loader = trainLoader[0]
            cnn1.load_state_dict(param.copy())
            for data in loader:
                image, label = data
                cnn1.train()
                optimizer1.zero_grad()
                output = cnn1.forward(image)
                loss = criterion(output, label)
                loss.backward()
                optimizer1.step()

        # save the current weight
        w[0] = cnn1.state_dict()

        # starts evaluation
        # cnn1.eval()
        # with torch.no_grad():
        #     for val_data in devLoader:
        #         image_val, label_val = val_data
        #         output_val = cnn1.forward(image_val)
        #         loss = criterion(output_val, label_val)
        #
        #         ps = torch.exp(output_val)
        #         top_p, top_class = ps.topk(1, dim=1)
        #         equals = top_class == label_val.view(*top_class.shape)
        #         accuracy_val[0] += torch.mean(equals.type(torch.FloatTensor)).item()
        #
        # # calculate accuracy for the current worker
        # accuracy_val[0] = accuracy_val[0] / len(devLoader)

        for epoch in range(inner_epochs):
            loader = trainLoader[1]
            cnn2.load_state_dict(param.copy())
            for data in loader:
                image, label = data
                cnn2.train()
                optimizer2.zero_grad()
                output = cnn2.forward(image)
                loss = criterion(output, label)
                loss.backward()
                optimizer2.step()

        # save the current weight
        w[1] = cnn2.state_dict()

        # starts evaluation
        # cnn2.eval()
        # with torch.no_grad():
        #     for val_data in devLoader:
        #         image_val, label_val = val_data
        #         output_val = cnn2.forward(image_val)
        #         loss = criterion(output_val, label_val)
        #
        #         ps = torch.exp(output_val)
        #         top_p, top_class = ps.topk(1, dim=1)
        #         equals = top_class == label_val.view(*top_class.shape)
        #         accuracy_val[1] += torch.mean(equals.type(torch.FloatTensor)).item()
        #
        # # calculate accuracy for the current worker
        # accuracy_val[1] = accuracy_val[1] / len(devLoader)


        for epoch in range(inner_epochs):
            loader = trainLoader[2]
            cnn3.load_state_dict(param.copy())
            for data in loader:
                image, label = data
                cnn3.train()
                optimizer3.zero_grad()
                output = cnn3.forward(image)
                loss = criterion(output, label)
                loss.backward()
                optimizer3.step()

        # save the current weight
        w[2] = cnn3.state_dict()

        # starts evaluation
        # cnn3.eval()
        # with torch.no_grad():
        #     for val_data in devLoader:
        #         image_val, label_val = val_data
        #         output_val = cnn3.forward(image_val)
        #         loss = criterion(output_val, label_val)
        #
        #         ps = torch.exp(output_val)
        #         top_p, top_class = ps.topk(1, dim=1)
        #         equals = top_class == label_val.view(*top_class.shape)
        #         accuracy_val[2] += torch.mean(equals.type(torch.FloatTensor)).item()
        #
        # # calculate accuracy for the current worker
        # accuracy_val[2] = accuracy_val[2] / len(devLoader)

        # calculate the corresponding weight for each worker
        # idx = accuracy_val / np.sum(accuracy_val)
        # weight1.append(idx[0])
        # weight2.append(idx[1])
        # weight3.append(idx[2])
        # print(idx)
        idx = [1/3,1/3,1/3]
        for key in param:
            # calculate the aggregated weight at the server side
            param[key] = idx[0]*w[0][key] + idx[1]*w[1][key] + idx[2]*w[2][key]


        # load the most up-to-date weight at the server for evaluation
        cnn.load_state_dict(param.copy())

        cnn.eval()
        total = 0
        correct = 0
        accuracy_test = 0
        accuracy_train = np.zeros(num_worker)

        test_loss = 0
        train_loss = np.zeros(num_worker)
        with torch.no_grad():

            # test the performance with the unseen data
            for val_data in testLoader:
                image_val, label_val = val_data
                output_val = cnn.forward(image_val)
                loss = criterion(output_val, label_val)
                test_loss += loss.item()

                ps = torch.exp(output_val)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == label_val.view(*top_class.shape)
                accuracy_test += torch.mean(equals.type(torch.FloatTensor)).item()


            # test the performance with the training images at each worker
            for i in range(len(trainLoader)):
                for val_data in trainLoader[i]:
                    image_val, label_val = val_data
                    output_val = cnn.forward(image_val)
                    loss = criterion(output_val, label_val)
                    train_loss[i] += loss.item()

                    ps = torch.exp(output_val)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == label_val.view(*top_class.shape)
                    accuracy_train[i] += torch.mean(equals.type(torch.FloatTensor)).item()

        loss_test.append(test_loss / len(testLoader))
        acc_test.append(100 * accuracy_test / len(testLoader))

        for i in range(num_worker):
            acc_train[i].append(100 * accuracy_train[i] / len(trainLoader[i]))
            loss_train[i].append(train_loss[i] / len(trainLoader[i]))
        print('epoch: {}, train 1 loss: {}, train 2 loss: {}, train 3 loss: {},test loss:{}'.format(
            outer, train_loss[0] / len(trainLoader[0]), train_loss[1] / len(trainLoader[1]),
            train_loss[2] / len(trainLoader[2]), test_loss / len(testLoader)))
        # cnn.train()

    # np.save('testacc_{}_worker_diff_5inner_avg'.format(num_worker), acc_test)
    np.save('testloss_{}_worker_diff_5inner_avg'.format(num_worker), loss_test)
    for i in range(num_worker):
        # np.save('trainacc_{}_worker_set_{}_diff_5inner_avg'.format(num_worker, i), acc_train[i])
        np.save('trainloss_{}_woker_set_{}_diff_5inner_avg'.format(num_worker,i), loss_train[i])
    # np.save('weight1',weight1)
    # np.save('weight2',weight2)
    # np.save('weight3',weight3)


