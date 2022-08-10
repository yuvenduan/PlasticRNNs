import math

import torch
import torch.nn as nn
import torch.nn.functional as F

fl = math.floor

__all__ = ['LeNet', 'ResNet']

class LeNet(nn.Module):
    def __init__(self, image_size, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.last_map_x = fl((fl((image_size[1]-4)/2)-4)/2)
        self.last_map_y = fl((fl((image_size[2]-4)/2)-4)/2)

        self.linear1 = nn.Linear(16 * self.last_map_x * self.last_map_y, 120)
        self.linear2 = nn.Linear(120, 84)
        self.out_layer = nn.Linear(84, num_classes)

    def forward(self, inp):
        x = self.pool1(F.relu(self.conv1(inp)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.last_map_x * self.last_map_y)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        otp = self.out_layer(x)
        return otp

    def set_mode(self, mod):
        pass

# class Adaptive_Batchnorm(nn.BatchNorm2d):

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# adapted from torchvision.models.resnet
# implemented using ResNet for CIFAR-10 setting in:
# Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition. arXiv:1512.03385
class ResNet(nn.Module):
    def __init__(self, layers, image_size, num_classes=10, zero_init_residual=False,
                 replace_stride_with_dilation=None, norm_layer=None, width=64):
        super(ResNet, self).__init__()

        if norm_layer is None or norm_layer == 'batchnorm':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instancenorm':
            norm_layer = nn.InstanceNorm2d
        elif norm_layer == 'none':
            norm_layer = nn.Identity
        else:
            raise NotImplementedError
        

        self._norm_layer = norm_layer

        self.inplanes = width // 4
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False]
        if len(replace_stride_with_dilation) != 2:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 2-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = nn.Conv2d(image_size[0], self.inplanes, kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, width // 4, layers[0])
        self.layer2 = self._make_layer(BasicBlock, width // 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(BasicBlock, width, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_layer = nn.Linear(width * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.out_layer(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def set_mode(self, mod):
        pass