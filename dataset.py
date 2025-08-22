import os
from glob import glob
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from config import Cfg

# ----- Data utilities -----
def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()

def remap_seg(seg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg, dtype=np.int64)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out

def zscore_norm(img: np.ndarray) -> np.ndarray:
    m = img[img > 0].mean() if (img > 0).any() else img.mean()
    s = img[img > 0].std() if (img > 0).any() else img.std()
    if s < 1e-8: 
        return img
    out = img.copy()
    mask = img > 0
    out[mask] = (img[mask] - m) / (s + 1e-8)
    return out

def crop_or_pad_3d(arr: np.ndarray, out_size: Tuple[int,int,int], center: Optional[Tuple[int,int,int]]=None) -> np.ndarray:
    D,H,W = arr.shape
    d,h,w = out_size
    if center is None:
        cz, cy, cx = D//2, H//2, W//2
    else:
        cz, cy, cx = center
    sz = max(0, min(D - d, cz - d//2))
    sy = max(0, min(H - h, cy - h//2))
    sx = max(0, min(W - w, cx - w//2))
    patch = arr[sz:sz+d, sy:sy+h, sx:sx+w]
    pdz = max(0, d - patch.shape[0])
    pdy = max(0, h - patch.shape[1])
    pdx = max(0, w - patch.shape[2])
    if pdz or pdy or pdx:
        patch = np.pad(patch, ((0,pdz),(0,pdy),(0,pdx)), mode='constant')
    return patch

def choose_tumor_center(seg_remap: np.ndarray) -> Optional[Tuple[int,int,int]]:
    idxs = np.argwhere(seg_remap > 0)
    if idxs.size == 0:
        return None
    z,y,x = idxs[np.random.randint(0, idxs.shape[0])]
    return int(z), int(y), int(x)

import os
from glob import glob
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from config import Cfg

# ----- Data utilities -----
def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()

def remap_seg(seg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg, dtype=np.int64)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out

def zscore_norm(img: np.ndarray) -> np.ndarray:
    m = img[img > 0].mean() if (img > 0).any() else img.mean()
    s = img[img > 0].std() if (img > 0).any() else img.std()
    if s < 1e-8: 
        return img
    out = img.copy()
    mask = img > 0
    out[mask] = (img[mask] - m) / (s + 1e-8)
    return out

def crop_or_pad_3d(arr: np.ndarray, out_size: Tuple[int,int,int], center: Optional[Tuple[int,int,int]]=None) -> np.ndarray:
    D,H,W = arr.shape
    d,h,w = out_size
    if center is None:
        cz, cy, cx = D//2, H//2, W//2
    else:
        cz, cy, cx = center
    sz = max(0, min(D - d, cz - d//2))
    sy = max(0, min(H - h, cy - h//2))
    sx = max(0, min(W - w, cx - w//2))
    patch = arr[sz:sz+d, sy:sy+h, sx:sx+w]
    pdz = max(0, d - patch.shape[0])
    pdy = max(0, h - patch.shape[1])
    pdx = max(0, w - patch.shape[2])
    if pdz or pdy or pdx:
        patch = np.pad(patch, ((0,pdz),(0,pdy),(0,pdx)), mode='constant')
    return patch

def choose_tumor_center(seg_remap: np.ndarray) -> Optional[Tuple[int,int,int]]:
    idxs = np.argwhere(seg_remap > 0)
    if idxs.size == 0:
        return None
    z,y,x = idxs[np.random.randint(0, idxs.shape[0])]
    return int(z), int(y), int(x)

class BraTSDataset(Dataset):
    def __init__(self, split=None, transform=None, eval_mode=False):
        self.transform = transform
        self.eval_mode = eval_mode

        if split is None:
            self.cases = sorted(glob(os.path.join(Cfg.data_dir, "*")))
        else:
            split_file = os.path.join(Cfg.split_dir, f"{split}.txt")
            if os.path.exists(split_file):
                with open(split_file) as f:
                    ids = [x.strip() for x in f.readlines()]
                self.cases = [os.path.join(Cfg.data_dir, id_) for id_ in ids]
            else:
                print(f"[WARNING] Split file {split_file} not found — using all data instead.")
                self.cases = sorted(glob(os.path.join(Cfg.data_dir, "*")))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        case_id = os.path.basename(case_dir)

        # Load modalities
        imgs = []
        for mod in Cfg.modalities:
            path = os.path.join(case_dir, f"{case_id}_{mod}.nii.gz")
            img = load_nifti(path)
            img = zscore_norm(img)
            imgs.append(img)
        img4d = np.stack(imgs, axis=0)  # (modalities, D, H, W)

        if self.eval_mode:
            return torch.from_numpy(img4d).float(), case_id

        # Load segmentation
        seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
        seg = load_nifti(seg_path) if os.path.exists(seg_path) else np.zeros_like(imgs[0])
        seg_remap = remap_seg(seg)

        return torch.from_numpy(img4d).float(), torch.from_numpy(seg_remap).long()
    
    def get_case_path(self, idx, modality="t1"):
        #--- Return the file path for a specific modality of the case."""
        case = self.cases[idx]
        case_id = os.path.basename(case)
        return os.path.join(case, f"{case_id}_{modality}.nii.gz")



"""

# ----- Dataset -----
class BraTSDataset(Dataset):
    # same content as your main.py class
    def __init__(self, split: str = "train"):
        self.split = split
        self.modalities = Cfg.modalities
        self.patch_size = Cfg.patch_size

        # Get cases
        split_file = os.path.join(Cfg.split_dir, f"{split}.txt")
        with open(split_file) as f:
            self.cases = [line.strip() for line in f]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        imgs = []
        for mod in self.modalities:
            path = os.path.join(Cfg.data_root, case, f"{case}_{mod}.nii.gz")
            img = load_nifti(path)
            img = zscore_norm(img)
            imgs.append(img)

        img4d = np.stack(imgs, axis=0)  # (4, D, H, W)

        # Segmentation (if available)
        seg_path = os.path.join(Cfg.data_root, case, f"{case}_seg.nii.gz")
        seg = load_nifti(seg_path) if os.path.exists(seg_path) else np.zeros_like(imgs[0])
        seg_remap = remap_seg(seg)

        # Patch extraction
        center = choose_tumor_center(seg_remap) if self.split == "train" else None
        img_patch = np.zeros((len(self.modalities), *self.patch_size), dtype=np.float32)
        seg_patch = np.zeros(self.patch_size, dtype=np.int64)
        for m in range(len(self.modalities)):
            img_patch[m] = crop_or_pad_3d(img4d[m], self.patch_size, center)
        seg_patch = crop_or_pad_3d(seg_remap, self.patch_size, center)

        return torch.from_numpy(img_patch), torch.from_numpy(seg_patch)
"""

        
class _EvalWrapper(Dataset):
    # same content as your main.py class
    """
    Wraps a dataset to only return images and paths (no labels).
    Useful for running predictions on full volumes.
    """
    def __init__(self, cases: list, modalities: list):
        self.cases = cases
        self.modalities = modalities

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        imgs = []
        for mod in self.modalities:
            path = os.path.join(Cfg.data_root, case, f"{case}_{mod}.nii.gz")
            img = load_nifti(path)
            img = zscore_norm(img)
            imgs.append(img)

        img4d = np.stack(imgs, axis=0)  # (modalities, D, H, W)
        return torch.from_numpy(img4d).float(), case



"""
# ----- Dataset -----
class BraTSDataset(Dataset):
    # same content as your main.py class
    def __init__(self, split: str = "train"):
        self.split = split
        self.modalities = Cfg.modalities
        self.patch_size = Cfg.patch_size

        # Get cases
        split_file = os.path.join(Cfg.split_dir, f"{split}.txt")
        with open(split_file) as f:
            self.cases = [line.strip() for line in f]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        imgs = []
        for mod in self.modalities:
            path = os.path.join(Cfg.data_root, case, f"{case}_{mod}.nii.gz")
            img = load_nifti(path)
            img = zscore_norm(img)
            imgs.append(img)

        img4d = np.stack(imgs, axis=0)  # (4, D, H, W)

        # Segmentation (if available)
        seg_path = os.path.join(Cfg.data_root, case, f"{case}_seg.nii.gz")
        seg = load_nifti(seg_path) if os.path.exists(seg_path) else np.zeros_like(imgs[0])
        seg_remap = remap_seg(seg)

        # Patch extraction
        center = choose_tumor_center(seg_remap) if self.split == "train" else None
        img_patch = np.zeros((len(self.modalities), *self.patch_size), dtype=np.float32)
        seg_patch = np.zeros(self.patch_size, dtype=np.int64)
        for m in range(len(self.modalities)):
            img_patch[m] = crop_or_pad_3d(img4d[m], self.patch_size, center)
        seg_patch = crop_or_pad_3d(seg_remap, self.patch_size, center)

        return torch.from_numpy(img_patch), torch.from_numpy(seg_patch)
"""

        
class _EvalWrapper(Dataset):
    # same content as your main.py class
    """
    Wraps a dataset to only return images and paths (no labels).
    Useful for running predictions on full volumes.
    """
    def __init__(self, cases: list, modalities: list):
        self.cases = cases
        self.modalities = modalities

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        imgs = []
        for mod in self.modalities:
            path = os.path.join(Cfg.data_root, case, f"{case}_{mod}.nii.gz")
            img = load_nifti(path)
            img = zscore_norm(img)
            imgs.append(img)

        img4d = np.stack(imgs, axis=0)  # (modalities, D, H, W)
        return torch.from_numpy(img4d).float(), case
