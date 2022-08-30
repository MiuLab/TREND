import torch
import pickle
import ipdb
from torch.utils.data import Dataset, DataLoader

class REDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, samples):
        batch = {}
        for key in ['input_ids', 'token_type_ids', 'label',\
                    'attention_mask', 'has_trigger', 'trigger_len']:
            batch[key] = [torch.tensor(sample[key], dtype=torch.long) for sample in samples]
            batch[key] = torch.vstack(batch[key])
        t_s = []
        t_e = []
        x_s = []
        x_e = []
        y_s = []
        y_e = []
        for sample in samples:
            t_s.append(sample['t_start'])
            t_e.append(sample['t_end'])
            x_s.append(sample['x_st'])
            x_e.append(sample['x_nd'])
            y_s.append(sample['y_st'])
            y_e.append(sample['y_nd'])
        t_idx = [t_s, t_e]
        x_idx = [x_s, x_e]
        y_idx = [y_s, y_e]
        batch['t_idx'] = torch.tensor(t_idx, dtype=torch.long).T
        batch['x_idx'] = torch.tensor(x_idx, dtype=torch.long).T
        batch['y_idx'] = torch.tensor(y_idx, dtype=torch.long).T


        return batch