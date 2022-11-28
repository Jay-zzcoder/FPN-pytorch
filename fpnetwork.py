""" Author: ZUOZUO
    Data: 2022.11.25
    Description: this is an implementation of ResNet50 with Feature Pyramid Structure"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchviz import make_dot
import timm
from torchsummary import summary


def con_bn_relu(in_channel,
                out_channel,
                kernel_size=1,
                stride=1,
                padding = 0
                ):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU())

def show(name, tensor):
    print(name+" shape: ", tensor.shape)




class residual_block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(residual_block, self).__init__()
        self.expansion = 4
        self.short_cut =True
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.stride = stride
        if self.stride ==2 or in_channels != out_channels*self.expansion:
            self.short_cut = False
            self.shortconv = nn.Conv2d(self.in_channel, self.out_channel*self.expansion, kernel_size=1, stride=self.stride)
        self.conv1 = con_bn_relu(self.in_channel, self.out_channel)
        self.conv2 = con_bn_relu(self.out_channel, self.out_channel, kernel_size=3, stride=self.stride, padding=1)
        self.conv3 = con_bn_relu(self.out_channel, self.out_channel*self.expansion)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.short_cut:
            out = out + x_
        else:
            x_ = self.shortconv(x_)
            #print("out", out.shape)
            #print("x_", x_.shape)
            out = out + x_
        out = self.relu(out)
        return out


class Resnet50_FPN(nn.Module):
    def __init__(self, input_size=640, output_channel=256, is_cls=False):
        super(Resnet50_FPN, self).__init__()
        num_blocks = [3, 4, 6, 3]
        self.is_cls = is_cls
        self.input_size = input_size
        self.output_channel = output_channel
        self.expansion = 4
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu1, self.maxpool)

        self.layer1 = self.make_layer(64 ,residual_block, num_blocks[0])
        self.layer2 = self.make_layer(128 ,residual_block, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(256 ,residual_block, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(512 ,residual_block, num_blocks[3], stride=2)
        if self.is_cls:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512*self.expansion, 1000)



    def make_layer(self, channel, block, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channel, channel, stride=stride))
        self.in_channel = channel * self.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)


    def forward(self, x):
        P = []
        out = self.layer1(self.layer0(x))
        c2 = out
        out = self.layer2(out)
        c3 = out
        out = self.layer3(out)
        c4 = out
        out = self.layer4(out)
        c5 = out

        if self.is_cls:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
        
        b5 = con_bn_relu(int(c5.shape[1]), self.output_channel).cuda()(c5)
        a5 = b5
        P5 = con_bn_relu(int(a5.shape[1]), self.output_channel, kernel_size=3, padding=1).cuda()(a5)
        avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        P5 = avg(P5)
        P5 = P5.squeeze(2)
        P5 = P5.squeeze(-1)
        P.append(P5)
        #show("P5", P5)
        
        b4 = con_bn_relu(int(c4.shape[1]), self.output_channel).cuda()(c4)
        a4 = b4 + F.interpolate(b5, (int(b4.shape[2]), int(b4.shape[2])), mode="nearest")
        P4 = con_bn_relu(int(a4.shape[1]), self.output_channel, kernel_size=3, padding=1).cuda()(a4)
        avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        P4 = avg(P4)
        P4 = P4.squeeze(2)
        P4 = P4.squeeze(-1)
        P.append(P4)
        #show("P4", P4)

        b3 = con_bn_relu(int(c3.shape[1]), self.output_channel).cuda()(c3)
        a3 = b3 + F.interpolate(b4, (int(b3.shape[2]), int(b3.shape[2])), mode="nearest")
        P3 = con_bn_relu(int(a3.shape[1]), self.output_channel, kernel_size=3, padding=1).cuda()(a3)
        avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        P3 = avg(P3)
        P3 = P3.squeeze(2)
        P3 = P3.squeeze(-1)
        P.append(P3)
        #show("P3", P3)

        b2 = con_bn_relu(int(c2.shape[1]), self.output_channel).cuda()(c2)
        a2 = b2 + F.interpolate(b3, (int(b2.shape[2]), int(b2.shape[2])), mode="nearest")
        P2 = con_bn_relu(int(a2.shape[1]), self.output_channel, kernel_size=3, padding=1).cuda()(a2)
        
        #P.append(P2)
        #show("P2", P2)
        
        return P

if __name__ == "__main__":

    #model = timm.create_model("resnet50", pretrained=True)
    x = torch.rand(1, 3, 640, 640)
    x = x.to("cuda")
    model = Resnet50_FPN(640, 256).cuda()
    y = model(x)
    
    y = torch.cat(y, dim=1)   
    print(len(y))
    print(y[0].shape)
    #summary(model, (3, 640, 640))

    """
    x=torch.rand(1,3,224,224)
    model = Resnet50(1000)
    y = model(x)
    g = make_dot(y)
    g.render('espnet_model.pdf', view=False)"""













































