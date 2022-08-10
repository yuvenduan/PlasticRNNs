__all__ = ['SeqReproduction']

import torch
import torch.utils.data as tud

from configs.configs import SeqReproductionConfig

class SeqReproduction(tud.Dataset):

    def __init__(self, config: SeqReproductionConfig):

        self.dim = config.input_shape[0] - 2
        self.seq_length = config.seq_length
        self.delay = config.delay

    def __len__(self):
        return 10000

    def __getitem__(self, index):

        seq = torch.rand((self.seq_length, self.dim)) * 2 - 1
        sample_input = torch.cat((
            seq,
            torch.ones((self.seq_length, 1)),
            torch.zeros((self.seq_length, 1)),
        ), dim=1)

        delay_input = torch.zeros((self.delay, self.dim + 2))
        test_input = torch.zeros((self.seq_length, self.dim + 2))
        test_input[:, -1] = 1

        input = torch.cat((sample_input, delay_input, test_input), dim=0)
        output = torch.cat((torch.zeros((self.seq_length + self.delay, self.dim)), seq), dim=0)

        return input, output