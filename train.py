import os.path as osp
import logging
import random

import numpy as np
import torch
import torch.optim.lr_scheduler as lrs

from configs.config_global import USE_CUDA, LOG_LEVEL, NP_SEED, TCH_SEED
from configs.configs import SupervisedLearningBaseConfig
from utils.logger import Logger
from datasets.data_sets import DatasetIters
from utils.train_utils import grad_clipping, model_init, task_init
from utils.config_utils import load_config

def train_from_path(path):
    """Train from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_train(config)

def model_eval(config, net, test_data, task_func, logger=None, eval=False, test=False):

    with torch.no_grad():
        net.eval()
        extra_out = []

        correct = 0
        total = 0
        test_loss = 0.0
        test_b = 0
        test_data.reset()
        
        for t_step_ in range(test_data.min_iter_len):
            loss_weighted, num_weighted, num_corr_weighted = 0, 0, 0
            for i_tloader, test_iter in enumerate(test_data.data_iters):
                mod_weight = config.mod_w[i_tloader]

                t_data = next(test_iter)
                result = task_func.roll(net, t_data, test=True)
                loss, num, num_corr = result[: 3]
                if config.do_analysis:
                    extra_out.append(result)

                loss *= mod_weight
                num *= mod_weight
                num_corr *= mod_weight

                loss_weighted += loss
                num_weighted += num
                num_corr_weighted += num_corr

            test_loss += loss_weighted
            total += num_weighted
            correct += num_corr_weighted

            test_b += 1
            if test_b >= config.test_batch:
                break

        if config.print_mode == 'accuracy':
            test_acc = 100 * correct / total
        else:
            test_error = correct / total

        avg_testloss = test_loss / test_b

        logger.log_tabular('TestLoss', avg_testloss)
        if config.print_mode == 'accuracy':
            logger.log_tabular('TestAcc', test_acc)
        else:
            logger.log_tabular('TestError', test_error)

        if hasattr(task_func, "after_validation_callback"):
            task_func.after_validation_callback(extra_out, logger, config.save_path)

        if config.print_mode == 'accuracy':
            return avg_testloss, test_acc
        else:
            return avg_testloss, test_error
        

def model_train(config: SupervisedLearningBaseConfig):
    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    random.seed(config.seed)

    assert config.config_mode == 'train', 'config mode must be train'
    if USE_CUDA:
        logging.info("training with GPU")

    # initialize network
    net = model_init(config, mode='train')

    # initialize logger
    logger = Logger(output_dir=config.save_path,
                    exp_name=config.experiment_name)

    if config.training_mode == 'RL':
        raise NotImplementedError

    # gradient clipping
    if config.grad_clip is not None:
        logging.info("Performs grad clipping with max norm " + str(config.grad_clip))

    # initialize task
    task_func = task_init(config)

    # initialize dataset
    train_data = DatasetIters(config, 'train', config.batch_size)

    if config.perform_val:
        test_data = DatasetIters(config, 'val', config.batch_size)

    # initialize optimizer
    if config.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.wdecay, amsgrad=True)
    elif config.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr,
                                    momentum=0.9, weight_decay=config.wdecay)
    else:
        raise NotImplementedError('optimizer not implemented')

    # initialize Learning rate scheduler
    if config.use_lr_scheduler:
        if config.scheduler_type == 'ExponentialLR':
            scheduler = lrs.ExponentialLR(optimizer, gamma=0.99)
        elif config.scheduler_type == 'StepLR':
            scheduler = lrs.StepLR(optimizer, 10, gamma=0.1)
        elif config.scheduler_type == 'CosineAnnealing':
            scheduler = lrs.CosineAnnealingLR(
                optimizer, 
                T_max=config.max_batch // config.log_every + 1,
                eta_min=config.lr / 10
            )
        else:
            raise NotImplementedError('scheduler_type must be specified')

    i_b = 0
    i_log = 0
    
    testloss_list = []
    testresult_list = []

    break_flag = False
    train_loss = 0.0

    for epoch in range(config.num_ep):
        train_data.reset()
        for step_ in range(train_data.min_iter_len):
            
            # save model
            if i_b % config.save_every == 0 and i_b > 0:
                torch.save(net.state_dict(), osp.join(config.save_path, 'net_{}.pth'.format(i_b)))
                if hasattr(task_func, "after_save_callback"):
                    task_func.after_save_callback(config, net, i_b)

            # log performance
            if i_b % config.log_every == 0:
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('BatchNum', i_b)
                logger.log_tabular('DataNum', i_b * config.batch_size * train_data.num_datasets)
                logger.log_tabular('TrainLoss', train_loss / config.log_every)
                
                if config.perform_val:
                    testloss, testresult = model_eval(config, net, test_data, task_func, logger, eval=True)
                    testloss_list.append(testloss)
                    testresult_list.append(testresult)

                    if (config.print_mode == 'accuracy' and testresult >= max(testresult_list)) or \
                        (config.print_mode == 'error' and testresult <= min(testresult_list)):
                        torch.save(net.state_dict(), osp.join(config.save_path, 'net_best.pth'))

                if i_b > 0 and config.use_lr_scheduler:
                    scheduler.step()
                
                logger.dump_tabular()

                train_loss = 0.0
                i_log += 1

            net.train()
            if hasattr(net, 'cnn') and config.freeze_pretrained_cnn and config.cnn_pretrain != 'none':
                net.cnn.eval()
            
            loss = 0.0
            optimizer.zero_grad()
            for i_loader, train_iter in enumerate(train_data.data_iters):
                mod_weight = config.mod_w[i_loader]
                data = next(train_iter)
                loss += task_func.roll(net, data, train=True) * mod_weight
            loss.backward()

            # gradient clipping
            if config.grad_clip is not None:
                grad_clipping(net, config.grad_clip)

            optimizer.step()
            train_loss += loss.item()

            i_b += 1
            if i_b >= config.max_batch + 1:
                break_flag = True
                break

        if break_flag:
            break

    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth')))

    if config.perform_test:

        # ensure the test set is the same for different random runs
        np.random.seed(NP_SEED)
        torch.manual_seed(TCH_SEED)
        random.seed(0)

        test_data = DatasetIters(config, 'test', config.batch_size)
        logger = Logger(output_dir=config.save_path, output_fname='test.txt', exp_name=config.experiment_name)

        model_eval(config, net, test_data, task_func, logger, test=True)
        logger.dump_tabular()

    if hasattr(task_func, "after_training_callback"):
        task_func.after_training_callback(config, net)
