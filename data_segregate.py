# data partition
#
# for single model:
# train_dir = 'train'
# dev_dir = 'dev'
# test_dir = 'test'
# dir_path = 'lfw'
# split function: use self.split()

# for federated learning where identities on each worker are the same
# train_dir = 'train1'
# dev_dir = 'train2'
# test_dir = 'train3'
# dir_path = 'train'
# split function: use self.split()

# for federated learning where identities on each worker are different
# train_dir = 'train1_diff'
# dev_dir = 'train2_diff'
# test_dir = 'train3_diff'
# dir_path = 'train'
# split function: use self.split_diff()


import os
import time
import torch
from shutil import copyfile
from torchvision import transforms

# train_dir = 'train'
# dev_dir = 'dev'
# test_dir = 'test'
# dir_path = 'lfw'

train_dir = 'train1_diff'
dev_dir = 'train2_diff'
test_dir = 'train3_diff'
dir_path = 'train'



class DataSegregate:
    def __init__(self, dir_path, train_dir, dev_dir, test_dir):
        self.labels = None
        self.dir_path = dir_path
        self.train_dir = train_dir
        self.dev_dir = dev_dir
        self.test_dir = test_dir
        self.images_dict = {}
        self.train_dict = {}
        self.dev_dict = {}
        self.test_dict = {}
        self.train_count = 0
        self.dev_count = 0
        self.test_count = 0
        self.total_count = 0
        self.total_label = 0
        self.make_dict()
        self.split_diff()
        # self.split()
        self.make_dirs()

    def make_dict(self):
        self.labels = os.listdir(self.dir_path)
        for label in self.labels:
            path = os.path.join(self.dir_path, label)
            images = os.listdir(path)
            count = len(images)
            self.images_dict[label] = images
            self.total_count += count

            ################ will need this part for single model case ##########################

            # if count >= 20 and count <= 30:
            #     self.images_dict[label] = images
            #     self.total_count += count
                # print(count)


    def split(self):
        for label, image_list in self.images_dict.items():
            num_images = len(image_list)
            hist = torch.histc(torch.range(0, num_images - 1), bins=2, min=0, max=num_images - 1)
            num_train, num_dev_test = hist.type(torch.LongTensor).tolist()
            num_train = int(num_images/3)
            num_dev = int(num_images/3)
            num_test = int(num_images/3)


            # make sure num_train = 80%, num_dev = num_test = 10%
            ############################ will need this line for single model partition ##############################
            # num_train, num_dev, num_test = self.even_split(num_train, num_dev_test)
            self.train_count += num_train
            self.dev_count += num_dev
            self.test_count += num_test
            self.train_dict[label] = image_list[:num_train]
            self.dev_dict[label] = image_list[num_train:num_train + num_dev]
            self.test_dict[label] = image_list[num_train + num_dev:]

    def split_diff(self):
        self.total_label = len(self.labels)
        num_train = int(self.total_label/3)
        num_dev = num_train
        num_test = num_train

        train_label = self.labels[:num_train]
        dev_label = self.labels[num_train:num_train + num_dev]
        test_label = self.labels[num_train + num_dev:]

        for label in self.labels:
            if label in train_label:
                img_list = self.images_dict[label]
                self.train_dict[label] = img_list
            else:
                self.train_dict[label] = []

            if label in dev_label:
                img_list = self.images_dict[label]
                self.dev_dict[label] = img_list
            else:
                self.dev_dict[label] = []

            if label in test_label:
                img_list = self.images_dict[label]
                self.test_dict[label] = img_list
            else:
                self.test_dict[label] = []

        # for label in train_label:
        #     img_list = self.images_dict[label]
        #     self.train_dict[label] = img_list
        #
        # for label in dev_label:
        #     img_list = self.images_dict[label]
        #     self.dev_dict[label] = img_list
        #
        # for label in test_label:
        #     img_list = self.images_dict[label]
        #     self.test_dict[label] = img_list


    def even_split(self, num_train, num_dev_test):
        while num_dev_test / (num_train + num_dev_test) > 0.2 and num_dev_test > 2:
            num_dev_test -= 1
            num_train += 1
        hist = torch.histc(torch.range(0, num_dev_test - 1), bins=2, min=0, max=num_dev_test - 1)
        num_dev, num_test = hist.type(torch.LongTensor).tolist()
        return num_train, num_dev, num_test

    def copy_files(self, data_dict, dir_output):
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        for label, image_list in data_dict.items():
            label_path_target = os.path.join(dir_output, label)
            label_path_source = os.path.join(self.dir_path, label)
            if not os.path.exists(label_path_target):
                os.makedirs(label_path_target)
            for image in image_list:
                source_path = os.path.join(label_path_source, image)
                target_path = os.path.join(label_path_target, image)
                copyfile(source_path, target_path)

    def make_dirs(self):
        self.copy_files(self.train_dict, train_dir)
        self.copy_files(self.dev_dict, dev_dir)
        self.copy_files(self.test_dict, test_dir)

hist = torch.histc(torch.Tensor(torch.range(0,1)), bins = 2, min=0, max=1)

tic = time.time()
data_seg = DataSegregate(dir_path, train_dir, dev_dir, test_dir)
toc = time.time()

print(data_seg.train_count/data_seg.total_count, data_seg.dev_count/data_seg.total_count, data_seg.test_count/data_seg.total_count)