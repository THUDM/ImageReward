'''
@File       :   utils.py
@Time       :   2023/01/14 22:49:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Some settings and tools.
'''

#encoding:utf-8
import os, shutil
import torch
from tensorboardX import SummaryWriter
from config.options import *
import torch.distributed as dist


""" ==================== Data ======================== """

def collate_fn(batch):
    return batch

def make_path():
    return "{}_bs{}_fix={}_lr={}{}".format(opts.savepath, opts.BatchSize, opts.fix_rate, opts.lr, opts.lr_decay_style)

""" ==================== Models ======================== """

def save_model(model):
    save_path = make_path()
    if not os.path.isdir(os.path.join(config['checkpoint_base'], save_path)):
        os.makedirs(os.path.join(config['checkpoint_base'], save_path), exist_ok=True)
    model_name = os.path.join(config['checkpoint_base'], save_path, 'best_lr={}.pt'.format(opts.lr))
    torch.save(model.state_dict(), model_name)


def load_model(model, ckpt_path = None):
    if ckpt_path is not None:
        model_name = ckpt_path
    else:
        load_path = make_path()
        if not os.path.isdir(os.path.join(config['checkpoint_base'], load_path)):
            os.makedirs(os.path.join(config['checkpoint_base'], load_path), exist_ok=True)
        model_name = os.path.join(config['checkpoint_base'], load_path, 'best_lr={}.pt'.format(opts.lr))
        
    print('load checkpoint from %s'%model_name)
    checkpoint = torch.load(model_name, map_location='cpu') 
    state_dict = checkpoint
    msg = model.load_state_dict(state_dict,strict=False)
    print("missing keys:", msg.missing_keys)

    return model 


def preload_model(model):

    state_dict = torch.load(opts.preload_path, map_location=model.device) 
    msg = model.load_state_dict(state_dict,strict=False)
    print("missing keys:", msg.missing_keys)

    return model 


""" ==================== Tools ======================== """

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777)

def visualizer():
    if get_rank() == 0:
        # filewriter_path = config['visual_base']+opts.savepath+'/'
        save_path = make_path()
        filewriter_path = os.path.join(config['visual_base'], save_path)
        if opts.clear_visualizer and os.path.exists(filewriter_path):   # 删掉以前的summary，以免重合
            shutil.rmtree(filewriter_path)
        makedir(filewriter_path)
        writer = SummaryWriter(filewriter_path, comment='visualizer')
        return writer
