



# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path
# import warnings
# import os

# class ChangeDataset(Dataset):
#     def __init__(self, rows, aug=None, ignore_index=255, normalize_imagenet=True, binary_mask=True):
#         self.rows = rows.reset_index(drop=True)
#         self.aug = aug
#         self.ignore_index = ignore_index
#         self.normalize_imagenet = normalize_imagenet
#         self.binary_mask = binary_mask

#         self._IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#         self._IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#     def __len__(self):
#         return len(self.rows)

#     def __getitem__(self, i):
#         r = self.rows.iloc[i]

#         # Debug prints and file existence asserts
#         # print(f"Loading t0_npy: {r.t0_npy}")
#         assert os.path.exists(r.t0_npy), f"t0_npy file missing: {r.t0_npy}"

#         # print(f"Loading t1_npy: {r.t1_npy}")
#         assert os.path.exists(r.t1_npy), f"t1_npy file missing: {r.t1_npy}"

#         # print(f"Loading mask_npy: {r.mask_npy}")
#         assert os.path.exists(r.mask_npy), f"mask_npy file missing: {r.mask_npy}"

#         t0 = np.load(r.t0_npy).astype(np.float32)
#         t1 = np.load(r.t1_npy).astype(np.float32)

#         if t0.ndim != 3 or t1.ndim != 3:
#             raise RuntimeError(f"Expected CHW arrays but got {t0.shape} and {t1.shape}")

#         x = np.concatenate([t0, t1], axis=0)

#         mask_path = Path(r.mask_npy)
#         mask = np.load(mask_path) if mask_path.suffix == '.npy' else np.array(Image.open(mask_path))
#         if mask.ndim == 3:
#             mask = mask[..., 0] if mask.shape[2] == 1 else mask[..., 0]

#         if self.aug:
#             C = x.shape[0] // 2
#             t0_hwc = np.transpose(x[:C, :, :], (1, 2, 0))
#             t1_hwc = np.transpose(x[C:, :, :], (1, 2, 0))
#             augmented = self.aug(image=t0_hwc, image_2=t1_hwc, mask=mask)
#             t0_hwc = augmented['image']
#             t1_hwc = augmented['image_2']
#             mask = augmented['mask']
#             t0 = np.transpose(t0_hwc, (2, 0, 1)).astype(np.float32)
#             t1 = np.transpose(t1_hwc, (2, 0, 1)).astype(np.float32)
#             x = np.concatenate([t0, t1], axis=0)
#             if mask.ndim == 3:
#                 mask = mask[..., 0] if mask.shape[2] == 1 else mask[..., 0]
#         else:
#             if mask.ndim == 3:
#                 mask = mask[..., 0]

#         if x.dtype == np.uint8 or x.max() > 2.0:
#             x = x.astype(np.float32) / 255.0

#         if self.normalize_imagenet:
#             n_ch = x.shape[0]
#             if n_ch % 3 == 0:
#                 n_trips = n_ch // 3
#                 for t in range(n_trips):
#                     for c in range(3):
#                         x[t*3 + c, :, :] = (x[t*3 + c, :, :] - self._IMG_MEAN[c]) / self._IMG_STD[c]
#             else:
#                 warnings.warn(f"Channel count {n_ch} not divisible by 3; skipping normalization")

#         mask = mask.astype(np.int64)
#         if self.binary_mask:
#             mask = np.where(mask == self.ignore_index, self.ignore_index, (mask > 0).astype(np.int64))

#         x = torch.from_numpy(x).float().contiguous()
#         y = torch.from_numpy(mask).long().contiguous()

#         return x, y

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import warnings
import os

class ChangeDataset(Dataset):
    def __init__(self, rows, base_dir=None, aug=None, ignore_index=255, normalize_imagenet=True, binary_mask=True):
        """
        Args:
            rows: pandas DataFrame with columns t0_npy, t1_npy, mask_npy, split
            base_dir: Base directory for resolving relative paths (e.g., 'data/LEVIR_CD/chips_256')
            aug: optional augmentation callable
            ignore_index: int, label value for ignored pixels (default 255)
            normalize_imagenet: bool, if True, apply ImageNet mean/std normalization
            binary_mask: bool, if True, map all positive classes >0 to 1
        """
        self.rows = rows.reset_index(drop=True)
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.aug = aug
        self.ignore_index = ignore_index
        self.normalize_imagenet = normalize_imagenet
        self.binary_mask = binary_mask

        self._IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _resolve_path(self, relative_path):
        """Resolve a relative path to absolute path using base_dir."""
        full_path = self.base_dir / relative_path
        return str(full_path.resolve())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]

        # Resolve paths relative to base directory
        t0_path = self._resolve_path(r.t0_npy)
        t1_path = self._resolve_path(r.t1_npy)
        mask_path = self._resolve_path(r.mask_npy)

        # Debug prints and file existence asserts
        assert os.path.exists(t0_path), f"t0_npy file missing: {t0_path} (from {r.t0_npy})"
        assert os.path.exists(t1_path), f"t1_npy file missing: {t1_path} (from {r.t1_npy})"
        assert os.path.exists(mask_path), f"mask_npy file missing: {mask_path} (from {r.mask_npy})"

        # Load files using resolved paths
        t0 = np.load(t0_path).astype(np.float32)
        t1 = np.load(t1_path).astype(np.float32)

        if t0.ndim != 3 or t1.ndim != 3:
            raise RuntimeError(f"Expected CHW arrays but got {t0.shape} and {t1.shape}")

        x = np.concatenate([t0, t1], axis=0)

        # Load mask
        mask_path_obj = Path(mask_path)
        mask = np.load(mask_path_obj) if mask_path_obj.suffix == '.npy' else np.array(Image.open(mask_path_obj))
        if mask.ndim == 3:
            mask = mask[..., 0] if mask.shape[2] == 1 else mask[..., 0]

        # Apply augmentations if any
        if self.aug:
            C = x.shape[0] // 2
            t0_hwc = np.transpose(x[:C, :, :], (1, 2, 0))
            t1_hwc = np.transpose(x[C:, :, :], (1, 2, 0))
            augmented = self.aug(image=t0_hwc, image_2=t1_hwc, mask=mask)
            t0_hwc = augmented['image']
            t1_hwc = augmented['image_2']
            mask = augmented['mask']
            t0 = np.transpose(t0_hwc, (2, 0, 1)).astype(np.float32)
            t1 = np.transpose(t1_hwc, (2, 0, 1)).astype(np.float32)
            x = np.concatenate([t0, t1], axis=0)
            if mask.ndim == 3:
                mask = mask[..., 0] if mask.shape[2] == 1 else mask[..., 0]
        else:
            if mask.ndim == 3:
                mask = mask[..., 0]

        # Normalize input images if values in 0..255 range
        if x.dtype == np.uint8 or x.max() > 2.0:
            x = x.astype(np.float32) / 255.0

        # ImageNet normalization applied separately per RGB triplet
        if self.normalize_imagenet:
            n_ch = x.shape[0]
            if n_ch % 3 == 0:
                n_trips = n_ch // 3
                for t in range(n_trips):
                    for c in range(3):
                        x[t*3 + c, :, :] = (x[t*3 + c, :, :] - self._IMG_MEAN[c]) / self._IMG_STD[c]
            else:
                warnings.warn(f"Channel count {n_ch} not divisible by 3; skipping normalization")

        # Convert mask to proper format
        mask = mask.astype(np.int64)
        if self.binary_mask:
            mask = np.where(mask == self.ignore_index, self.ignore_index, (mask > 0).astype(np.int64))

        # Convert to torch tensors
        x = torch.from_numpy(x).float().contiguous()
        y = torch.from_numpy(mask).long().contiguous()

        return x, y

    def get_sample_info(self, i):
        """Return info dict about specific sample for debugging."""
        r = self.rows.iloc[i]
        mask_path = self._resolve_path(r.mask_npy)
        
        mask_path_obj = Path(mask_path)
        mask = np.load(mask_path_obj) if mask_path_obj.suffix == '.npy' else np.array(Image.open(mask_path_obj))

        if mask.ndim == 3:
            mask = mask[..., 0] if mask.shape[2] == 1 else mask[..., 0]

        unique_vals = np.unique(mask)
        pos_pixels = int(((mask == 1) & (mask != self.ignore_index)).sum())
        total_pixels = int((mask != self.ignore_index).sum())
        positive_ratio = float(pos_pixels / max(total_pixels, 1))

        return {
            'index': i,
            'positive_pixels': pos_pixels,
            'total_valid_pixels': total_pixels,
            'positive_ratio': positive_ratio,
            'unique_values': unique_vals.tolist(),
            'files': {
                't0': self._resolve_path(r.t0_npy),
                't1': self._resolve_path(r.t1_npy),
                'mask': self._resolve_path(r.mask_npy)
            }
        }