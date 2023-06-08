
import os
import json
import math
import torch
from torch.utils.data import Dataset
from config.utils import *
from config.options import *
from rank_pair_dataset import rank_pair_dataset

if __name__ == "__main__":
    train_dataset = rank_pair_dataset("train")
    valid_dataset = rank_pair_dataset("valid")
    test_dataset = rank_pair_dataset("test")
    
    train_dataset.store_dataset("train")
    valid_dataset.store_dataset("valid")
    test_dataset.store_dataset("test")