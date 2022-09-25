import os
import os.path as osp
import torch
import logging

from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as transform
from configs.configs import BaseConfig, SupervisedLearningBaseConfig
from datasets.visual_datasets import ContrastiveLearningViewGenerator, CIFAR10ImgVariationGenerator, get_simclr_pipeline_transform
import datasets

from configs.config_global import ROOT_DIR


# TODO: could implement this entire structure as an iterator,
# so that a batch is a list of batches for each dataloader
# to consider reset all iterator at the end of training and testing
# an alternative could be a data iterator that keep iterating though all datasets
# note that the notion of epoch doesn't applies anymore
class DatasetIters(object):

    def __init__(self, config, phase, b_size):
        """
        Initialize a list of data loaders
        if only one dataset is specified, return a list containing one data loader
        """
        if type(config.dataset) is list or type(config.dataset) is tuple:
            dataset_list = config.dataset
        elif type(config.dataset) is str:
            dataset_list = [config.dataset]
        else:
            raise NotImplementedError('Dataset config not recognized')

        self.num_datasets = len(dataset_list)
        assert len(config.mod_w) == self.num_datasets, 'mod_w and dataset len must match'

        self.data_iters = []
        self.iter_lens = []
        self.min_iter_len = None

        self.data_loaders = []
        for d_set in dataset_list:
            data_loader = init_single_dataset(d_set, phase, b_size, config)
            self.data_loaders.append(data_loader)
        self.reset()

    def reset(self):
        # recreate iterator for each of the dataset
        self.data_iters = []
        self.iter_lens = []
        for data_l in self.data_loaders:
            data_iter = iter(data_l)
            self.data_iters.append(data_iter)
            self.iter_lens.append(len(data_iter))
        self.min_iter_len = min(self.iter_lens)

def init_single_dataset(dataset_name, phase, b_size, config: SupervisedLearningBaseConfig=None):
    collate_f = None
    num_wks = config.num_workers
    train_flag = phase == 'train'

    if dataset_name == 'MNIST':
        trans = transform.Compose([transform.ToTensor(), ])
        # transform.Normalize((0.1307,), (0.3081,))])

        data_set = torchvision.datasets.MNIST(root=osp.join(ROOT_DIR, 'data'),
                                              train=train_flag, download=True,
                                              transform=trans)

    elif dataset_name == 'CIFAR10':
        # classes = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
        #            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        # trans = transform.Compose([transform.ToTensor(),
        #                            transform.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        trans = transform.ToTensor()
        data_set = torchvision.datasets.CIFAR10(root=osp.join(ROOT_DIR, 'data'),
                                                train=train_flag, download=True,
                                                transform=trans)

    elif dataset_name == 'CueReward':
        data_set = datasets.CueReward(config)

    elif dataset_name == 'SeqReproduction':
        data_set = datasets.SeqReproduction(config)

    elif dataset_name == 'Regression':
        data_set = datasets.Regression(config)

    elif dataset_name == 'FSC':
        dataset = config.image_dataset

        if dataset == 'miniImageNet':
            from datasets.fsc.mini_imagenet import MiniImageNet, FewShotDataloader
            dataset = MiniImageNet(phase=phase)
            data_loader = FewShotDataloader
        elif dataset == 'tieredImageNet':
            from datasets.fsc.tiered_imagenet import tieredImageNet, FewShotDataloader
            dataset = tieredImageNet(phase=phase)
            data_loader = FewShotDataloader
        elif dataset == 'CIFAR_FS':
            from datasets.fsc.CIFAR_FS import CIFAR_FS, FewShotDataloader
            dataset = CIFAR_FS(phase=phase)
            data_loader = FewShotDataloader
        elif dataset == 'FC100':
            from datasets.fsc.FC100 import FC100, FewShotDataloader
            dataset = FC100(phase=phase)
            data_loader = FewShotDataloader
        else:
            print ("Cannot recognize the dataset type")
            assert(False)
        
        if train_flag:
            data_loader = data_loader(
                dataset=dataset,
                nKnovel=config.train_way,
                nKbase=0,
                nExemplars=config.train_shot, # num training examples per novel category
                nTestNovel=config.train_way * config.train_query, # num test examples for all the novel categories
                nTestBase=0, # num test examples for all the base categories
                batch_size=b_size,
                num_workers=num_wks,
                epoch_size=b_size * 1000, # num of batches per epoch
            )
        else:
            data_loader = data_loader(
                dataset=dataset,
                nKnovel=config.train_way,
                nKbase=0,
                nExemplars=config.train_shot, # num training examples per novel category
                nTestNovel=config.train_query * config.train_way, # num test examples for all the novel categories
                nTestBase=0, # num test examples for all the base categories
                batch_size=b_size,
                num_workers=0,
                epoch_size=b_size * config.test_batch, # num of batches per epoch
            )
        data_loader = data_loader

    elif dataset_name == 'Image':

        dataset = config.image_dataset

        if dataset == 'miniImageNet':
            from datasets.fsc.mini_imagenet import MiniImageNet, FewShotDataloader
            dataset = MiniImageNet(phase='train')
        elif dataset == 'tieredImageNet':
            from datasets.fsc.tiered_imagenet import tieredImageNet, FewShotDataloader
            dataset = tieredImageNet(phase='train')
        elif dataset == 'CIFAR_FS':
            from datasets.fsc.CIFAR_FS import CIFAR_FS, FewShotDataloader
            dataset = CIFAR_FS(phase='train')
        elif dataset == 'FC100':
            from datasets.fsc.FC100 import FC100, FewShotDataloader
            dataset = FC100(phase='train')
        else:
            print ("Cannot recognize the dataset type")
            assert(False)

        total_length = len(dataset)
        indices = []
        for idx in range(total_length):
            if (phase == 'train') == (idx % 5 > 0):
                indices.append(idx)
        data_set = Subset(dataset, indices)

    elif dataset_name == 'SimCLR':

        dataset = config.image_dataset

        if dataset == 'miniImageNet':
            from datasets.fsc.mini_imagenet import MiniImageNet, FewShotDataloader
            dataset = MiniImageNet(phase='train', do_not_use_random_transf=True)
            size = 84
        elif dataset == 'tieredImageNet':
            from datasets.fsc.tiered_imagenet import tieredImageNet, FewShotDataloader
            dataset = tieredImageNet(phase='train', do_not_use_random_transf=True)
        elif dataset == 'CIFAR_FS':
            from datasets.fsc.CIFAR_FS import CIFAR_FS, FewShotDataloader
            dataset = CIFAR_FS(phase='train', do_not_use_random_transf=True)
            size = 32
        elif dataset == 'FC100':
            from datasets.fsc.FC100 import FC100, FewShotDataloader
            dataset = FC100(phase='train', do_not_use_random_transf=True)
        else:
            print ("Cannot recognize the dataset type")
            assert(False)

        dataset.transform = ContrastiveLearningViewGenerator(
            get_simclr_pipeline_transform(size, normalize=dataset.normalize)
        )
        data_set = dataset

    else:
        raise NotImplementedError('Dataset not implemented')

    if dataset_name != 'FSC':
        data_loader = DataLoader(data_set, batch_size=b_size, shuffle=train_flag,
                             num_workers=num_wks, drop_last=True, collate_fn=collate_f)
                            
    return data_loader
