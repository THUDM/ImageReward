'''
@File       :   rank_dataset.py
@Time       :   2023/01/20 15:38:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Rank dataset for reward model.
'''

import os
import json
import math
from torch.utils.data import Dataset
from config.options import *

class rank_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset_path = os.path.join(config['data_base'], f"{dataset}.json")
        with open(self.dataset_path, "r") as f:
            self.data = json.load(f)
        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
