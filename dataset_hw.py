from torch.utils.data import Dataset, DataLoader
import time
import math
import numpy as np
import os
import torch
import cv2
# import Augment
import random

class LineGenerate():
    # def __init__(self, IAMPath, conH, conW, augment = False, training=False):
    #     self.training = training
    #     self.augment = augment
    #     self.conH = conH
    #     self.conW = conW
    #     standard = []
        # with open(IAMPath) as f:
        #     for line in f.readlines():
        #         standard.append(line.strip('\n'))
        # self.image = []
        # self.label = []
        # line_prefix = '/'.join(IAMPath.split('/')[:-1]) + '/lines'
        # IAMLine = line_prefix + '.txt'
        # count = 0
        # with open(IAMLine) as f:
        #     for line in f.readlines():
        #         elements = line.split()
        #         pth_ele = elements[0].split('-')
        #         line_tag = '%s-%s' % (pth_ele[0], pth_ele[1])
        #         if line_tag in standard:
        #             pth = line_prefix + '/%s/%s-%s/%s.png' % (pth_ele[0], pth_ele[0], pth_ele[1], elements[0])
        #             img= cv2.imread(pth, 0) #see channel and type
        #             self.image.append(img)
        #             self.label.append(elements[-1])
        #             count += 1

    # def __init__(self, IAMPath, conH, conW, augment=False, training=True):
    #     self.training = training  # training=False
    #     self.augment = augment  # augment = False数据扩增
    #     self.conH = conH  # height
    #     self.conW = conW  # weight
    #     standard = []
    #     self.image = []
    #     self.label = []
    #     count = 0
    #     with open(IAMPath, encoding='gb18030') as f:  # IAMPath: Hw chemical formula/image_list.txt
    #         for line in f.readlines():
    #             line = line.strip('\n').split(" label: ")  # [99-O_3.jpg,O_3]  [= (14).jpg,~]
    #             # print(line)
    #             # train_data_path = "".join(line[0:-1]).split("/")[1]  # [27-NaHCO_3.jpg]
    #             # if ("→(" in train_data_path) or ("+(" in train_data_path) or ("=(" in train_data_path) or ("#(" in train_data_path):
    #             #     train_data_path = " (".join(train_data_path.split("("))
    #             # print(train_data_path)
    #             self.train_data_path = line[0]
    #             standard.append(self.train_data_path)  # [27-NaHCO_3.jpg,-30-13-Hg(NO_3)_2.jpg,....]
    #             train_label = line[-1]    #是-1
    #             # if "eval" in IAMPath:
    #             #     print("test labels",self.train_data_path,train_label)
    #             # else:
    #             #     print("train labels", self.train_data_path, train_label)
    #             self.label.append(train_label)
    #             try:
    #                 # img = cv2.imdecode(np.fromfile(os.path.join("E:/Guzhiwen_project/Hw_chemical_formula/train_data",train_data_path), dtype=np.uint8), -1)
    #                 img = cv2.imread(os.path.join("E:/Guzhiwen_project/Hw_chemical_formula/train_data",self.train_data_path),0)
    #                 img = cv2.resize(img,(0,0),fx=3,fy=3)    #对输入resize
    #                 count += 1
    #                 self.image.append(img)
    #                 try:
    #                     h, w = img.shape
    #                 except:
    #                     print("'NoneType' object image path:",self.train_data_path)
    #             except:
    #                 print("train_data_path:", self.train_data_path)
    #     self.len = count
    #     self.idx = 0

    def __init__(self, IAMPath, conH, conW, augment=False, training=True):
        self.training = training  # training=False
        self.augment = augment  # augment = False数据扩增
        self.conH = conH  # height
        self.conW = conW  # weight
        standard = []
        self.image = []
        self.label = []
        count = 0
        with open(IAMPath, encoding='gb18030') as f:  # IAMPath: Hw chemical formula/image_list.txt
            for line in f.readlines():
                line = line.strip('\n').split(" label: ")  # [99-O_3.jpg,O_3]  [= (14).jpg,~]
                self.train_data_path = line[0]
                standard.append(self.train_data_path)  # [27-NaHCO_3.jpg,-30-13-Hg(NO_3)_2.jpg,....]
                train_label = line[-1]    #是-1
                # if "eval" in IAMPath:
                #     print("test labels",self.train_data_path,train_label)
                # else:
                #     print("train labels", self.train_data_path, train_label)
                try:
                    # img = cv2.imdecode(np.fromfile(os.path.join("E:/Guzhiwen_project/Hw_chemical_formula/train_data",train_data_path), dtype=np.uint8), -1)
                    img = cv2.imread(os.path.join("E:/Guzhiwen_project/Hw_chemical_formula/train_data",self.train_data_path),0)
                    img = cv2.resize(img,(0,0),fx=3,fy=3)    #对输入resize
                    self.label.append(train_label)
                    count += 1
                    self.image.append(img)
                    try:
                        h, w = img.shape
                    except:
                        print("'NoneType' object image path:",self.train_data_path)
                except:
                    print("train_data_path:", self.train_data_path)
        self.len = count
        self.idx = 0

    def get_len(self):
        return self.len

    def generate_line(self):
        if self.training:
            idx = np.random.randint(self.len)
            image = self.image[idx]
            label = self.label[idx]
            # print('print(self.label[idx])',self.label[idx])
        else:
            idx = self.idx
            image = self.image[idx]
            label = self.label[idx]
            # print('print(self.label[idx])', self.label[idx])
            self.idx += 1
        if self.idx == self.len:
            self.idx -= self.len

        h,w = image.shape
        imageN = np.ones((self.conH,self.conW))*255
        beginH = int(abs(self.conH-h)/2)
        beginW = int(abs(self.conW-w)/2)
        if h <= self.conH and w <= self.conW:
            imageN[beginH:beginH+h, beginW:beginW+w] = image
        elif float(h) / w > float(self.conH) / self.conW:
            newW = int(w * self.conH / float(h))
            beginW = int(abs(self.conW-newW)/2)
            image = cv2.resize(image, (newW, self.conH))
            imageN[:,beginW:beginW+newW] = image
        elif float(h) / w <= float(self.conH) / self.conW:
            newH = int(h * self.conW / float(w))
            beginH = int(abs(self.conH-newH)/2)
            image = cv2.resize(image, (self.conW, newH))
            imageN[beginH:beginH+newH] = image
        label = self.label[idx]

        # if self.augment and self.training:
        #     imageN = imageN.astype('uint8')
        #     if torch.rand(1) < 0.3:
        #         imageN = Augment.GenerateDistort(imageN, random.randint(3, 8))
        #     if torch.rand(1) < 0.3:
        #         imageN = Augment.GenerateStretch(imageN, random.randint(3, 8))
        #     if torch.rand(1) < 0.3:
        #         imageN = Augment.GeneratePerspective(imageN)

        imageN = imageN.astype('float32')
        imageN = (imageN-127.5)/127.5
        return imageN, label




class WordGenerate():
    def __init__(self, IAMPath, conH, conW, augment = False):
        self.augment = augment
        self.conH = conH
        self.conW = conW
        standard = []
        with open(IAMPath) as f:
            for line in f.readlines():
                standard.append(line.strip('\n'))
        self.image = []
        self.label = []

        word_prefix = '/'.join(IAMPath.split('/')[:-1]) + '/words'
        IAMWord = word_prefix + '.txt'
        count = 0
        with open(IAMWord) as f:
            for line in f.readlines():
                elements = line.split()
                pth_ele = elements[0].split('-')
                line_tag = '%s-%s' % (pth_ele[0], pth_ele[1])
                if line_tag in standard:
                    pth = word_prefix + '/%s/%s-%s/%s.png' % (pth_ele[0], pth_ele[0], pth_ele[1], elements[0])
                    img= cv2.imread(pth, 0) #see channel and type
                    if img is not None:
                        self.image.append(img)
                        self.label.append(elements[-1])
                        count += 1
                    else:
                        print('error')
                        continue;

        self.len = count

    def get_len(self):
        return self.len

    def word_generate(self):

        endW = np.random.randint(50);
        label = ''
        imageN = np.ones((self.conH,self.conW))*255
        imageList =[]
        while True:
            idx = np.random.randint(self.len)
            image = self.image[idx]
            h,w = image.shape
            beginH = int(abs(self.conH-h)/2)
            imageList.append(image)
            if endW + w > self.conW:
                break;
            if h <= self.conH:
                imageN[beginH:beginH+h, endW:endW+w] = image
            else:
                imageN[:,endW:endW+w] = image[beginH:beginH+self.conH]

            endW += np.random.randint(60)+20+w
            if label == '':
                label = self.label[idx]
            else:
                label = label + '|' + self.label[idx]
                print("word_label",label)

        label = label
        imageN = imageN.astype('uint8')
        # if self.augment:
        #     if torch.rand(1) < 0.3:
        #         imageN = Augment.GenerateDistort(imageN, random.randint(3, 8))
        #     if torch.rand(1) < 0.3:
        #         imageN = Augment.GenerateStretch(imageN, random.randint(3, 8))
        #     if torch.rand(1) < 0.3:
        #         imageN = Augment.GeneratePerspective(imageN)

        imageN = imageN.astype('float32')
        imageN = (imageN-127.5)/127.5
        return imageN, label

class IAMDataset(Dataset):
    def __init__(self, img_list, img_height, img_width, transform=False):
        IAMPath = img_list
        self.conH = img_height
        self.conW = img_width
        self.LG = LineGenerate(IAMPath, self.conH, self.conW)

    def __len__(self):
        return self.LG.get_len()

    def __getitem__(self, idx):

        imageN, label = self.LG.generate_line()

        imageN = imageN.reshape(1,self.conH,self.conW)
        sample = {'image': torch.from_numpy(imageN), 'label': label}

        return sample

class IAMSynthesisDataset(Dataset):
    def __init__(self, img_list, img_height, img_width, augment = False, transform=None):
        self.training = True
        self.augment = augment
        IAMPath = img_list
        self.conH = img_height
        self.conW = img_width
        self.LG = LineGenerate(IAMPath, self.conH, self.conW, self.augment, self.training)
        self.WG = WordGenerate(IAMPath, self.conH, self.conW, self.augment)

    def __len__(self):
        return self.WG.get_len()

    def __getitem__(self, idx):
        if np.random.rand() < 0.5:
            imageN, label = self.LG.generate_line()
        else:
            imageN, label = self.WG.word_generate()

        imageN = imageN.reshape(1,self.conH,self.conW)
        sample = {'image': torch.from_numpy(imageN), 'label': label}
        return sample



