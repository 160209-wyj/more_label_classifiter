import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn



class GCNResnet_junjie(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file='data/voc/voc_adj.pkl'):
        super(GCNResnet_junjie, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        # self.pooling = nn.MaxPool2d(14, 14)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.relu = nn.LeakyReLU(0.2)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature):
        feature = self.features(feature)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature= self.fc(feature)
        return feature

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr},
                # {'params': self.gc2.parameters(), 'lr': lr},
                ]




def gcn_resnet50(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    return GCNResnet_junjie(model, num_classes)