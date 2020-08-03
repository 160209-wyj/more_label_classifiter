import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image,ImageStat,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
import util
from util import *
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,ImageCompression
)

object_categories = ['公章', '显示器', '投影仪', '红头文件',  #4
                     '正常文件', '工程图纸','正常']

def strong_aug(p=0.5):
    return Compose([
        ImageCompression(quality_lower=60,quality_upper=80,compression_type=0,always_apply=True,p=0.2),
        RandomRotate90(p=0.2),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(p=1),
            GaussNoise(p=1)
        ], p=0.2),
        OneOf([
            MotionBlur(p=1),
            MedianBlur(blur_limit=3, p=1),
            Blur(blur_limit=3, p=1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=360, p=0.2),
        OneOf([
            OpticalDistortion(p=1),
            GridDistortion(p=1),
            IAAPiecewiseAffine(p=1),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
                RandomBrightnessContrast(),
        ], p=0.1),
        HueSaturationValue(p=0.2),
    ], p=p)

def read_object_labels_csv_2_three_class(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    porn_class=0
    sexy_class=1
    normal_class=2
    porn_count=0
    sexy_count=0
    normal_count=0
    with open(file, 'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.int)
                if labels[15]==1 and sum(labels)==1:
                    labels=normal_class
                    normal_count+=1
                elif sum(labels[0:16])>1:
                    labels=porn_class
                    porn_count+=1
                elif sum(labels[16:18])>1:
                    labels=sexy_class
                    sexy_count+=1
                elif sum(labels[18:])>1:
                    normal_count+=1
                    labels=normal_class
                else:
                    labels=normal_class
                    normal_count+=1

                # labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
        print("porn: ",porn_count,"sexy: ",sexy_count,"normal: ",normal_count)
    return images

def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)

    with open(file, 'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.int)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
        return images
 
def read_object_txt(file, header=True):
    images = []
    print('[dataset] read', file)
    with open(file, 'r',encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip('\n')
            images.append(data)
    return images


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images

class PornClassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
        self.root = root
        self.path_images = self.root
        self.set = set
        self.transform = transform

        self.target_transform = target_transform

        # define filename of csv file
        file_csv = os.path.join('/home/zhangjunjie/work/junjie/ML-GCN/data/bk/after_yanshi_1_csv',set+'.csv')
        if set=='train':
            self.aug=strong_aug(p=0.5)
        else:
            self.aug=None
        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)     
        
        print('[dataset] Porn classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))
 
    def __getitem__(self, index):
        path, target = self.images[index]
        # path=int(path)
        # folder='porn'+ str(path//10000)
        # image_name="/%06d"% (path) + '.jpg'
        
        # folder=path
        image_name=path
        image_path='/home/zhangjunjie/work/junjie/all_new_add_normal_resize/'+image_name
        # img = Image.open(os.path.join(self.path_images, image_path)).convert('RGB')
        img = Image.open(image_path).convert('RGB')

        # stat=ImageStat.Stat(img)
        # img_mean=np.array(stat.mean)/255
        # img_std=np.array(stat.stddev)/255
        # normalize=transforms.Normalize(mean=img_mean,std=img_std)

        #  data augmentation
        if self.aug is not None:
            # print(self.aug)
            # Convert PIL image to numpy array
            image_np = np.array(img)
            # Apply transformationss
            augmented = self.aug(image=image_np)
            # Convert numpy array to PIL Image
            img = Image.fromarray(augmented['image'])
            # img.save('/home/zhangchao/1.jpg')

        if self.transform is not None:
            # transform_use=transforms.Compose(self.transform)
            img=self.transform(img)
            # img=transform_use(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            print('target:',target)
        return (img, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)


class PornClassificationTest(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
        self.root = root
        self.path_images = self.root
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # define filename of csv file
        # file_csv = 'output_5w.txt'
        file_csv='100w.txt'

        self.classes = object_categories
        self.images = read_object_txt(file_csv)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] Porn classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        image_path  = self.images[index]
        img = Image.open(os.path.join(self.path_images, image_path)).convert('RGB')

        target = (np.zeros([21,])).astype(np.float32)
        target = torch.from_numpy(target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, image_path, self.inp), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)


class PornClassification_strong(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
        self.root = root
        self.path_images = self.root
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # define filename of csv file
        file_csv = os.path.join('./data/porn/strong_label.csv')

        self.classes = object_categories
        # self.images = read_object_labels_csv(file_csv)
        images = read_object_labels_csv(file_csv)
        # shuffle default True
        train_data, test_data = train_test_split(images, test_size=0.2, random_state=1)
# , y_train, y_test

        if set=='train':
            self.images=train_data
        elif set=='val':
            self.images=test_data
        print('[dataset] Porn classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))
    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, path), target
    def __len__(self):
            return len(self.images)
    def get_number_classes(self):
            return len(self.classes)
