import os
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transform

from configs.config_global import ROOT_DIR

# code to make WhiteNoise dataset
# WhiteNoise is a dataset that contain 10000 white noise images
# Cropped to img_size
def make_white_noise_dataset(data_path, img_size=(3, 32, 32), image_num=10000):
    print("Making WhiteNoise dataset")
    channel, height, width = img_size
    for i_ in tqdm(range(image_num)):
        img_data = np.random.rand(height, width, channel)
        img_data = np.uint8(img_data * 255)
        img = Image.fromarray(img_data)
        # img.show()

        image_path = os.path.join(data_path, str(i_))
        # If folder doesn't exist, then create it.
        if not os.path.isdir(image_path):
            os.makedirs(image_path)

        fig_name = (str(i_) + '.jpg')
        img.save(os.path.join(image_path, fig_name))


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class GaussianBlur(object):
    """
    blur a single image on CPU
    adapted from: https://github.com/sthalles/SimCLR
    """
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transform.ToTensor()
        self.tensor_to_pil = transform.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def get_simclr_pipeline_transform(size, s=1, normalize=nn.Identity()):
    """
    Return a set of data augmentation transformations as described in the SimCLR paper.
    adapted from: https://github.com/sthalles/SimCLR
    """
    color_jitter = transform.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transform.Compose([transform.RandomResizedCrop(size=size),
                                         transform.RandomHorizontalFlip(),
                                         transform.RandomApply([color_jitter], p=0.8),
                                         transform.RandomGrayscale(p=0.2),
                                         GaussianBlur(kernel_size=int(0.1 * size)),
                                         transform.ToTensor(),
                                         normalize])
    return data_transforms


class CIFAR10ImgVariationGenerator(object):

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x)


class ShufflePixel(object):
    r"""
    get white noise by shuffling pixels of each img independently
    adapted from: https://github.com/sthalles/SimCLR
    """
    def __init__(self):
        self.pil_to_tensor = transform.ToTensor()
        self.tensor_to_pil = transform.ToPILImage()
    
    def __call__(self, img):
        # img size = (channel, width, height)

        img = self.pil_to_tensor(img).unsqueeze(0)
        img_shape = img.shape
        img_shape = [i for i in img_shape]
        resize_shape = img_shape[:-2]
        resize_shape.append(-1)
        img = img.view(resize_shape)
        idx = torch.randperm(img.shape[-1])
        img = img[:, :, idx]
        img = img.view(img_shape)
        img = img.squeeze(0)

        img = self.tensor_to_pil(img)

        return img


def get_shuffle_pixel_pipeline_transform():
    r"""
    Return a set of data of shuffled pixels.
    adapted from: https://github.com/sthalles/SimCLR
    """
    data_transforms = transform.Compose([ShufflePixel(),
                                         transform.ToTensor()])
    return data_transforms