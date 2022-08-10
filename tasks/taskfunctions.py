import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp

from configs.config_global import DEVICE
import torchvision.transforms as transform
from configs.configs import BaseConfig, ContrastivePretrainConfig, FSCConfig

from utils.logger import Logger

def data_batch_to_device(data_b, device=DEVICE):
    if type(data_b) is torch.Tensor:
        return data_b.to(device)
    elif type(data_b) is tuple or type(data_b) is list:
        return [data_batch_to_device(data, device=device) for data in data_b]
    else:
        raise NotImplementedError("input type not recognized")

class ContrastiveLearning:
    
    def __init__(self, config: ContrastivePretrainConfig):
        self.batch_size = config.batch_size
        self.temperature = 0.07
        self.criterion = torch.nn.CrossEntropyLoss()

    def info_nce_loss(self, features):
        # adapted from https://github.com/sthalles/SimCLR
        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(DEVICE)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (2 * self.batch_size, 2 * self.batch_size)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

        logits = logits / self.temperature
        return logits, labels

    def roll(self, model, data_batch, train=False, test=False, evaluate=False):
        assert train + test + evaluate == 1, "only one mode should be activated"
        input_, label_ = data_batch_to_device(data_batch)

        input_ = torch.cat(input_, dim=0)
        features = model(input_)
        logits, labels = self.info_nce_loss(features)
        task_loss = self.criterion(logits, labels)

        if train:
            return task_loss
        else:
            raise NotImplementedError("Not Implemented")\

    def after_save_callback(self, config, model, idx):
        torch.save(model.cnn, osp.join(config.save_path, f'cnn_{idx}.pth'))

class Classification:

    def __init__(self, config):
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = config.batch_size
        self.task_batch_s = self.batch_size

    def roll(self, model, data_batch, train=False, test=False, evaluate=False):
        assert train + test + evaluate == 1, "only one mode should be activated"
        input_, label_ = data_batch_to_device(data_batch)
        output = model(input_)

        task_loss = self.criterion(output, label_)

        pred_num = 0
        pred_correct = 0

        if test or evaluate:
            _, pred_id = torch.max(output.detach(), 1)
            pred_num += label_.size(0)
            pred_correct += (pred_id == label_).sum().item()

        if train:
            return task_loss
        elif test or evaluate:
            return task_loss.item(), pred_num, pred_correct
        else:
            raise NotImplementedError("Not Implemented")

    def after_save_callback(self, config, model, idx):
        torch.save(model.cnn, osp.join(config.save_path, f'cnn_{idx}.pth'))

class SeqRegression:

    def __init__(self, config):

        self.criterion = nn.MSELoss()
        self.batch_size = config.batch_size
        self.test_begin = config.test_begin

    def roll(self, model, data_batch, train=False, test=False, evaluate=False):
        assert train + test + evaluate == 1, "only one mode should be activated"
        
        input, target = data_batch_to_device(data_batch)

        pred_num = 0
        sum_error = 0
        task_loss = 0
        
        hidden = torch.zeros((input.shape[0], model.memory_size), device=DEVICE)

        for t, (x, y) in enumerate(zip(input.unbind(1), target.unbind(1))):
            out, hidden = model(x, hidden)
            if t >= self.test_begin:
                task_loss = task_loss + self.criterion(out, y)

        task_loss = task_loss / (input.shape[1] - self.test_begin)
        sum_error = task_loss.item() * input.shape[0]
        pred_num = input.shape[0]

        if train:
            return task_loss
        elif test or evaluate:
            return task_loss.item(), pred_num, sum_error
        else:
            raise NotImplementedError("Not Implemented")

class FSC:

    def __init__(self, config: FSCConfig):

        self.criterion = nn.CrossEntropyLoss()
        
        self.way = config.train_way
        self.shot = config.train_shot
        self.query = config.train_query

    def roll(self, model, data_batch, train=False, test=False, evaluate=False):
        assert train + test + evaluate == 1, "only one mode should be activated"
        
        img_support, labels_support, img_query, labels_query, _, _ = data_batch_to_device(data_batch)
        onehot_support = F.one_hot(labels_support, num_classes=self.way)

        pred_num = 0
        correct_num = 0
        task_loss = 0
        
        bsz = img_support.shape[0]
        hidden = torch.zeros((bsz, model.memory_size), device=DEVICE)

        for img, input in zip(img_support.unbind(1), onehot_support.unbind(1)):
            _, hidden = model((img, input), hidden)

        for img, label in zip(img_query.unbind(1), labels_query.unbind(1)):
            out, hidden = model((img, torch.zeros_like(onehot_support[:, 0])), hidden)
            task_loss += self.criterion(out, label)

            correct_num += (torch.argmax(out, dim=-1) == label).sum().item()
            pred_num += bsz

        task_loss = task_loss / img_query.shape[1]

        if train:
            return task_loss
        elif test or evaluate:
            return task_loss.item(), pred_num, correct_num
        else:
            raise NotImplementedError("Not Implemented")