import os
import random
import numpy as np
import torch

class Cfg:
    # Paths
    data_root = r"D:\Ahmad\SuperLightNet\archive\BraTS2021_Training_Data"  # main dataset folder
    data_dir  = data_root  # ✅ Alias so older code using `Cfg.data_dir` still works
    ckpt_dir  = r"D:\Ahmad\SuperLightNet\checkpoints"
    split_dir = r"D:\Ahmad\SuperLightNet\splits"  # folder with train.txt / val.txt

    # Logging
    log_every = 1

    # Data processing
    modalities = ["t1", "t1ce", "t2", "flair"]
    patch_size = (128, 128, 128)
    samples_per_volume = 2
    tumor_bias = 0.7
    intensity_norm = "z"

    # Model / Training
    num_classes = 4
    epochs = 100
    batch_size = 1
    lr = 1e-3
    weight_decay = 1e-5
    amp = True
    num_workers = 4
    train_val_split = 0.9
    seed = 42

    # RMD settings
    rmd_enable = True
    rmd_min_keep = 2
    rmd_prob = 0.7

    # Sliding window inference
    sw_input_size = (160, 160, 160)
    sw_overlap = 0.5

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(Cfg.seed)
