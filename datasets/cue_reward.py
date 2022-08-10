__all__ = ['CueReward']

import torch
import torch.utils.data as tud

from configs.configs import CueRewardConfig

class CueReward(tud.Dataset):

    def __init__(self, config: CueRewardConfig):

        self.n_class = config.n_class
        self.noise = config.input_noise
        self.trial_length = config.trial_length
        self.input_size = config.input_shape[0]

    def __len__(self):
        return 10000

    def __getitem__(self, index):

        cls = torch.randint(0, self.n_class, size=(self.trial_length, )) 
        stimuli = torch.rand((self.n_class, self.input_size - 2))
        reward = torch.rand((self.n_class, )) * 2 - 1

        input = torch.empty((self.trial_length, self.input_size))
        output = torch.empty((self.trial_length, 1))

        for i in range(self.trial_length // 2):
            input[i][: -2] = stimuli[cls[i]] + torch.randn_like(stimuli[cls[i]]) * self.noise
            input[i][-2] = reward[cls[i]]
            input[i][-1] = 0
            output[i][0] = reward[cls[i]]

        for i in range(self.trial_length // 2, self.trial_length):
            input[i][: -2] = stimuli[cls[i]] + torch.randn_like(stimuli[cls[i]]) * self.noise
            input[i][-2] = 0
            input[i][-1] = 1
            output[i][0] = reward[cls[i]]

        return input, output