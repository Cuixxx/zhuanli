import torch
import os
import torch.nn as nn

from torchvision import models
import numpy as np
import random
os.environ['TORCH_HOME'] = 'models'#指定预训练模型下载地址

alexnet_model = models.alexnet(pretrained=True)
# print(alexnet_model)
# #


class SpacialNet(nn.Module):
    def __init__(self):
        super(SpacialNet, self).__init__()
        self.features = nn.Sequential(*list(alexnet_model.features.children()))

        self.FC = nn.Sequential(nn.Linear(9216, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU())

    def forward(self, x):
        #空间特征
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)

        return x

class SpacialNet_Pan(nn.Module):
    def __init__(self):
        super(SpacialNet_Pan, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1, stride=1, padding=0)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, padding=0)
        )
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size = 3, stride = 1, padding=1),
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(2, stride=2,padding=0),#128*128*64
                                      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(2, stride=2, padding=0),#64*64*128
                                      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(2, stride=2, padding=0),#32*32*256
                                      self.block1,
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(2, stride=2, padding=0),# 16*16*512
                                      self.block2,
                                      nn.LeakyReLU(0.2, True),
                                      nn.AvgPool2d(16, stride=1, padding=0)# 1*1*1024
                                      )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze()
        return x




class gf1mulNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Spacial_Net = SpacialNet()
        self.Spectral_Net1 = Spectral_Net()
        self.mixlayer = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU())
        self.hash = nn.Sequential(nn.Linear(1024, 64), nn.Tanh()).cuda()

    def forward(self, img):#img1 GF1 mul,img2 GF2 mul,img3 GF1 pan

        spacial_feat = self.Spacial_Net(img[:,0:3,:,:])
        spectral_feat = self.Spectral_Net1(img)
        mix_feat = self.mixlayer(torch.cat((spacial_feat, spectral_feat), dim=1))

        hash_code = self.hash(mix_feat)
        return mix_feat, hash_code#(-1,1)

class gf2mulNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Spacial_Net = SpacialNet()
        self.Spectral_Net2 = Spectral_Net()
        self.mixlayer = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU())
        self.hash = nn.Sequential(nn.Linear(1024, 64), nn.Tanh()).cuda()

    def forward(self, img):#img1 GF1 mul,img2 GF2 mul,img3 GF1 pan

        spacial_feat = self.Spacial_Net(img[:,0:3,:,:])
        spectral_feat = self.Spectral_Net2(img)
        mix_feat = self.mixlayer(torch.cat((spacial_feat, spectral_feat), dim=1))

        hash_code = self.hash(mix_feat)
        return mix_feat, hash_code#(-1,1)

class gf1panNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.PAN_Net = SpacialNet_Pan()
        self.mixlayer = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU())
        self.hash = nn.Sequential(nn.Linear(1024, 64), nn.Tanh()).cuda()

    def forward(self, img):#img1 GF1 mul,img2 GF2 mul,img3 GF1 pan


        pan_feat = self.PAN_Net(img)

        hash_code = self.hash(pan_feat)
        return pan_feat, hash_code#(-1,1)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Spacial_Net = SpacialNet()
        self.PAN_Net = SpacialNet_Pan()
        self.hash = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 64), nn.Tanh()).cuda()
        self.MergeLayer_gf1 = nn.Sequential(nn.Conv2d(4, 3, 1, 1, 0), nn.LeakyReLU(0.2, True))
        self.MergeLayer_gf2 = nn.Sequential(nn.Conv2d(4, 3, 1, 1, 0), nn.LeakyReLU(0.2, True))

    def forward(self, img1, img2, img3):#img1 GF1 mul,img2 GF2 mul,img3 GF1 pan
        Merge_feat1 = self.MergeLayer_gf1(img1)
        Mul_feat1 = self.Spacial_Net(Merge_feat1)

        Merge_feat2 = self.MergeLayer_gf2(img2)
        Mul_feat2 = self.Spacial_Net(Merge_feat2)

        pan_feat = self.PAN_Net(img3)

        cat_vector = torch.cat((Mul_feat1, Mul_feat2, pan_feat), dim=0)
        hash_code = self.hash(cat_vector)
        return cat_vector, hash_code#(-1,1)

    # def forward(self, spacial_img, spectral_vector, img_pan, state):
    #     if state == 0: #训练
    #         spacial_vector = self.net1(spacial_img)
    #         mixed_vector = self.net2(spectral_vector, spacial_vector)
    #         pan_vector = self.net3(img_pan)
    #         cat_vector = torch.cat((mixed_vector, pan_vector), dim=0)
    #         hash_code = self.hash(cat_vector)
    #     elif state == 1: #校验输入mul
    #         spacial_vector = self.net1(spacial_img)
    #         mixed_vector = self.net2(spectral_vector, spacial_vector)
    #         hash_code = self.hash(mixed_vector)
    #     elif state == 2:#校验输入pan
    #         pan_vector = self.net3(img_pan)
    #         hash_code = self.hash(pan_vector)
    #     return hash_code#(-1,1)


if __name__ == '__main__':
    model = MyModel()
    print(model)