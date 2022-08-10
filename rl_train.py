import torch
import torch_ac as ac
import torch.nn as nn
import gym
import logging
import time
import numpy as np
import os.path as osp
import torchvision.transforms as trans

from configs.configs import RLBaseConfig

from utils.logger import Logger
from configs.config_global import DEVICE

# Set run dir
def reshape_reward(obs, action, reward, done):
    if not done:
        reward = -1
    else:
        reward = 1
    return reward

def rl_train(config: RLBaseConfig, model: nn.Module, logger: Logger):

    # Load environments
    envs = []
    for i in range(config.num_envs):
        env = gym.make(config.env, **config.env_kwargs)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.horizon)
        env.seed(config.seed + 10000 * i)
        envs.append(env)
        
    logging.info("Environments loaded\n")

    def transform(obs, device):
        image_transform = trans.Compose([
            trans.ToPILImage(),
            trans.Resize((210, 160)),
            trans.ToTensor()
        ])
        obs = [image_transform(ob) for ob in obs]
        obs = torch.stack(obs)
        return obs.to(device)

    # Load algo
    if config.algo == "a2c":
        algo = ac.A2CAlgo(
            envs, model, DEVICE, 
            lr=config.lr, 
            num_frames_per_proc=config.horizon,
            entropy_coef=config.entropy_coef,
            recurrence=config.recurrence,
            max_grad_norm=config.grad_clip,
            preprocess_obss=transform
        )

    elif config.algo == "ppo":
        algo = ac.PPOAlgo(
            envs, model, DEVICE, 
            lr=config.lr, 
            num_frames_per_proc=config.horizon,
            entropy_coef=config.entropy_coef,
            recurrence=config.recurrence,
            max_grad_norm=config.grad_clip,
            batch_size=config.batch_size,
            epochs=config.num_ep,
            clip_eps=config.clip_epsilon,
            preprocess_obss=transform
        )
    else:
        raise ValueError("Incorrect algorithm name: {}".format(config.algo))

    # Train model
    num_frames = 0
    update = 0
    start_time = time.time()
    best_return = -1e9
    all_logs = []

    while update < config.max_batch:

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()
        num_frames += logs["num_frames"]
        update += 1
        # Print logs

        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = np.mean(logs["return_per_episode"])
        # rreturn_per_episode = np.mean(logs["reshaped_return_per_episode"])
        num_frames_per_episode = np.mean(logs["num_frames_per_episode"])

        header = ["return_per_episode", "num_frames_per_episode"]
        data = [return_per_episode, num_frames_per_episode]

        header += ["FPS", ]
        data += [fps, ]
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
        all_logs.append(data)

        if update % config.log_every == 0:
            
            for idx, key in enumerate(header):
                data[idx] = np.mean([(
                    log[idx].item() if isinstance(log[idx], torch.Tensor) else log[idx]
                ) for log in all_logs])

            header += ['duration', 'update', "frames"]
            data += [duration, update, num_frames]

            for key, value in zip(header, data):
                logger.log_tabular(key, value)

            logger.dump_tabular()
            all_logs = []

            if data[0] > best_return:
                best_return = return_per_episode
                torch.save(model.state_dict(), osp.join(config.save_path, 'net_best.pth'.format(update)))

        # Save status

        if update % config.save_every == 0:
            torch.save(model.state_dict(), osp.join(config.save_path, 'net_{}.pth'.format(update)))
            logging.info("Model saved")