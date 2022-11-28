import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

def con_bn_relu(in_channel,
                out_channel,
                kernel_size=1,
                stride=1,
                padding = 0
                ):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU(inplace=True))


class resnet50_fpn(nn.Module):
    def __init__(self,  input_size=640, output_channel=256):
        super(resnet50_fpn, self).__init__()
        
        self.output_channel = output_channel
        self.resnet = timm.create_model("resnet50",
                                        pretrained=True,
                                        features_only=True)

    def forward(self, x):
        P = []

        y = self.resnet(x)
        c5 = y[4]
        c4 = y[3]
        c3 = y[2]
        c2 = y[1]
        
        b5 = con_bn_relu(int(c5.shape[1]), self.output_channel).cuda()(c5)
        a5 = b5
        P5 = con_bn_relu(int(a5.shape[1]), self.output_channel, kernel_size=3, padding=1).cuda()(a5)
        P5 = b5
        P.append(P5)
        # show("P5", P5)

        
        b4 = con_bn_relu(int(c4.shape[1]), self.output_channel).cuda()(c4)
        a4 = b4 + F.interpolate(b5, (int(b4.shape[2]), int(b4.shape[2])), mode="nearest")
        P4 = con_bn_relu(int(a4.shape[1]), self.output_channel, kernel_size=3, padding=1).cuda()(a4)
        P.append(P4)
        # show("P4", P4)

        b3 = con_bn_relu(int(c3.shape[1]), self.output_channel).cuda()(c3)
        a3 = b3 + F.interpolate(b4, (int(b3.shape[2]), int(b3.shape[2])), mode="nearest")
        P3 = con_bn_relu(int(a3.shape[1]), self.output_channel, kernel_size=3, padding=1).cuda()(a3)

        P.append(P3)
        # show("P3", P3)
        

        b2 = con_bn_relu(int(c2.shape[1]), self.output_channel).cuda()(c2)
        a2 = b2 + F.interpolate(b3, (int(b2.shape[2]), int(b2.shape[2])), mode="nearest")
        P2 = con_bn_relu(int(a2.shape[1]), self.output_channel, kernel_size=3, padding=1).cuda()(a2)
        P.append(P2)
        # show("P2", P2)

        return P



if __name__ == "__main__":
    x = torch.rand(1, 3, 640, 640)
    model = resnet50_fpn()
    y = model(x)
    print(y[0].shape)