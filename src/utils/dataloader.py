import os
import torch
import numpy as np

from configs.pretrain import TrainingConfig

def _load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ('train', 'val')

        data_root = TrainingConfig().data_root
        shards = os.listdir(data_root)
        shards = sorted([s for s in shards if split in s])
        shards = [os.path.join(data_root, shard) for shard in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        if self.process_rank == 0:
            print(f"Found {len(shards)} for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = _load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        buf = self.tokens[self.current_position:self.current_position + (self.B * self.T) + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T * self.num_processes
        if self.current_position + (self.B * self.T * self.num_processes) + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = _load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y