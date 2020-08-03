from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import tqdm
import torchvision.models as models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 占用GPU90%的显存
#session = tf.Session(config=config)

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
   


plt.ion()   # interactive mode

# 图片路径
# save_path = '/home/guomin/.cache/torch/checkpoints/resnet18-customs-angle.pth'
model_path = '/home/user/junjie/ML-GCN/models/checkpoint.pth'
 
# ------------------------ 加载数据 --------------------------- #
# Data augmentation and normalization for training
# Just normalization for validation
# 定义预训练变换
preprocess_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
 
class_names = ['公章', '显示器', '投影仪', '红头文件',  
                     '正常文件', '工程图纸']
# class_names = ['阴茎', '外阴', '下体', '乳房',
#                      '臀部', '性交', '给女性口交', '给男性口交', '给女性手交',
#                      '给男性手交', '无性器官性交动作', '无性器官口交动作',
#                      '足交', '乳交', '道具',
#                      '颜射', '比基尼', '短裙', '拥抱', '亲吻','自拍']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# ------------------------ 载入模型并且训练 --------------------------- #
num_classes = 6
model = gcn_resnet50(num_classes=num_classes,pretrained=True)
checkpoint  = torch.load(model_path)
model.load_state_dict(checkpoint ['state_dict'])
model.eval()
# print(model)

# dire = '/home/zhangjunjie/keras-applications-master/kafka_data/all_test_data/'
# dire = '/home/zhangjunjie/MobileNetV2-master/image_140_yuanchicun/images_highrisk/'

# path = os.listdir(dire)
# for file in tqdm.tqdm(path):

image_PIL = Image.open('100000340.jpg')
if image_PIL.mode != 'RGB':
    # print('image:',image_PIL,'image_path:',dire+file)
    image_PIL = image_PIL.convert("RGB")
    # os.remove(dire+file)
    # image_PIL.save(dire+file)
#
image_tensor = preprocess_transform(image_PIL)
# 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
image_tensor.unsqueeze_(0)
# 没有这句话会报错
image_tensor = image_tensor.to(device)
model = model.to('cuda')
out = model(image_tensor)
## 4. 标签和分数输出
prob = torch.sigmoid(out).squeeze()
# preds = [(class_names[i], prob[i].item()) for i,score in enumerate(prob) if score>0.5]
preds = [(class_names[i], prob[i].item()) for i,score in enumerate(prob)]
# [标签，分数]的列表
print(preds)
    # 得到预测结果，并且从大到小排序
    # _, indices = torch.sort(out, descending=True)
    # # 返回每个预测值的百分数
    # percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    # max_val =percentage.argmax()
    
    # print(class_names[max_val], percentage[max_val].item())

    # save_path = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/dy_pytorch_save_data'
    # save_path = '/home/zhangjunjie/MobileNetV2-master/img_crop'
    # def read():
    #     if percentage[max_val].item() >= 70:
    #         img = cv2.imread(dire+file)
    #         cv2.imwrite('{}/{}/{}.jpg'.format(save_path,class_names[max_val],percentage[max_val].item()),img)
    # if max_val == 0:
    #     read() 
    # if max_val == 1:
    #     read()
    # if max_val == 2:
    #     read()
    # if max_val == 3:
    #     read()
    # if max_val == 4:
    #     read()
    # if max_val == 5:
    #     read()
    # if max_val == 6:
    #     read()
    # if max_val == 7:
    #     read()
    # if max_val == 8:
    #     read()
    # if max_val == 9:
    #     read()
    # if max_val == 10:
    #     read()
    # if max_val == 11:
    #     read()
    # if max_val == 12:
    #     read()
    # if max_val == 13:
    #     read()
    # if max_val == 14:
    #     read()
    # if max_val == 15:
    #     read()
    # if max_val == 16:
    #     read()
    # if max_val == 17:
    #     read()
    # if max_val == 18:
    #     read()
    # if max_val == 19:
    #     read()
    # if max_val == 20:
    #     read()
    




