import os
import math
import time
import torch
from torch.utils.data import DataLoader, random_split
from config import Cfg
from dataset import BraTSDataset, _EvalWrapper
from main import MultiEncoderRMDUNet
from loss_metrics import DiceCELoss, dice_per_region

def make_loaders():
    full = BraTSDataset(
        root=Cfg.data_root,
        modalities=Cfg.modalities,
        patch_size=Cfg.patch_size,
        samples_per_volume=Cfg.samples_per_volume,
        tumor_bias=Cfg.tumor_bias,
        intensity_norm=Cfg.intensity_norm,
        train=True
    )
    n_total = len(full)
    n_train = int(n_total * Cfg.train_val_split)
    n_val = n_total - n_train
    train_set, val_set = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(Cfg.seed))
    val_set = _EvalWrapper(val_set, root=Cfg.data_root, modalities=Cfg.modalities)

    train_loader = DataLoader(train_set, batch_size=Cfg.batch_size, shuffle=True,
                              num_workers=Cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,
                              num_workers=Cfg.num_workers, pin_memory=True)
    print(f"Dataset: total={n_total}, train={n_train}, val={n_val}")
    return train_loader, val_loader

def train():
    os.makedirs(Cfg.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = make_loaders()
    model = MultiEncoderRMDUNet(
        in_modalities=len(Cfg.modalities),
        base_ch=16,
        num_stages=4,
        num_classes=Cfg.num_classes,
        rmd_enable=Cfg.rmd_enable
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=Cfg.amp)
    criterion = DiceCELoss(num_classes=Cfg.num_classes)

    start_epoch = 1
    best_val = math.inf
    latest_ckpt = os.path.join(Cfg.ckpt_dir, "latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", math.inf)

    for epoch in range(start_epoch, Cfg.epochs + 1):
        # ---- Training loop exactly as in main.py ----
        ...
