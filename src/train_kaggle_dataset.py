# train_kaggle_dataset.py
import os
import glob
import math
import time
import csv
import pickle
import random
import shutil
import argparse
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================
SEED = 42
ROOT = "./data/kaggle"         # scans under here; supports subfolders like 'jpeg', 'cxy', 'masks', etc.
IMG_SIZE = 512                 # resize side (square)
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3
VAL_SPLIT = 0.15
OUT_DIR = "./model/kaggle/out_seg"
WEIGHTS_PATH = os.path.join(OUT_DIR, "transunet.pt")
INDEX_PKL = os.path.join(OUT_DIR, "index.pkl")
SPLIT_CSV = os.path.join(OUT_DIR, "split.csv")
LOG_CSV = os.path.join(OUT_DIR, "train_log.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPORT_TS_PATH = os.path.join(OUT_DIR, "transunet.ts")

# Workers (Windows can hang with many)
NUM_WORKERS_TRAIN = 2
NUM_WORKERS_VAL = 1

# default radius used when only .cxy is present and no radius file is found
DEFAULT_RAD_PX = 24

# file type hints
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_DIR_HINTS = ("mask", "masks", "roi", "rois", "ground_truth", "gt", "annotation", "annotations", "seg")
MASK_NAME_HINTS = ("mask", "_mask", "-mask", "roi", "_roi", "-roi", "seg", "_seg", "-seg", "gt", "_gt", "-gt")
SIDE_EXTS = {".txt", ".csv", ".cxy", ".rad"}

# Use CBIS-DDSM CSV-defined splits instead of random split
USE_CSV_SPLITS = True

# Dataset selection and pathology filtering
# DATASETS: tuple containing any of ('mass', 'calc')
DATASETS: Tuple[str, ...] = ("mass", "calc")
# PATHOLOGY_FILTER: None for all, or one of {"MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"}
PATHOLOGY_FILTER: Optional[str] = None

# =========================
# Helpers
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def is_image_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS

def pil_to_tensor_image(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=2)
    arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(arr)

def pil_to_tensor_mask(msk: Image.Image) -> torch.Tensor:
    arr = np.array(msk).astype(np.float32)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    arr = (arr > 127).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr)

def resize_img(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR)

def resize_mask(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.NEAREST)

def is_mask_like(path: str) -> bool:
    p = path.lower().replace("\\", "/")
    name = os.path.basename(p)
    return any(h in p for h in MASK_DIR_HINTS) or any(h in name for h in MASK_NAME_HINTS)

def stem_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def parse_numbers_from_file(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline().strip()
    for sep in [",", ";", " ", "\t", "|"]:
        if sep in line:
            parts = [p for p in line.split(sep) if p]
            break
    else:
        parts = [line]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except Exception:
            continue
    return nums

# =========================
# Fast index (single scan) + cache
# =========================
def build_index_fast(root: str) -> List[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    """
    Single pass over ROOT:
      - collect images
      - collect mask-like images
      - collect .cxy (under 'cxy' folder) and .rad (under 'rad' or 'radius')
    Returns: list of (image_path, mask_image_or_None, cxy_or_None, rad_or_None)
    """
    print("[index] scanning files...")
    t0 = time.time()
    all_paths = glob.glob(os.path.join(root, "**", "*.*"), recursive=True)
    t1 = time.time()
    print(f"[index] found {len(all_paths)} files in {t1 - t0:.2f}s")

    images: List[str] = []
    mask_imgs: List[str] = []
    cxy_files: List[str] = []
    rad_files: List[str] = []

    for p in all_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in IMG_EXTS:
            if is_mask_like(p):
                mask_imgs.append(p)
            else:
                images.append(p)
        elif ext in SIDE_EXTS:
            lp = p.lower().replace("\\", "/")
            if "cxy" in lp and ext in {".txt", ".csv", ".cxy"}:
                cxy_files.append(p)
            elif ("rad" in lp or "radius" in lp) and ext in {".txt", ".csv", ".rad"}:
                rad_files.append(p)

    # build maps
    def map_by_stem(paths: List[str]) -> Dict[str, List[str]]:
        m: Dict[str, List[str]] = {}
        for q in paths:
            m.setdefault(stem_no_ext(q).lower(), []).append(q)
        return m

    mask_map = map_by_stem(mask_imgs)
    cxy_map = map_by_stem(cxy_files)
    rad_map = map_by_stem(rad_files)

    print(f"[index] images: {len(images)} | mask_imgs: {len(mask_imgs)} | cxy: {len(cxy_files)} | rad: {len(rad_files)}")

    pairs: List[Tuple[str, Optional[str], Optional[str], Optional[str]]] = []
    missing = 0
    for ip in images:
        s = stem_no_ext(ip).lower()
        # mask exact or with suffix
        hits = mask_map.get(s, [])
        if not hits:
            for suf in MASK_NAME_HINTS:
                hits = mask_map.get(s + suf, [])
                if hits:
                    break
        if hits:
            hits_sorted = sorted(
                hits,
                key=lambda pth: (
                    ("mask" not in os.path.basename(pth).lower()),
                    not any(h in pth.lower() for h in MASK_DIR_HINTS),
                    pth.lower(),
                ),
            )
            pairs.append((ip, hits_sorted[0], None, None))
            continue

        # no mask image; try .cxy
        cxy = None
        rad = None
        if s in cxy_map:
            cxy = cxy_map[s][0]
        else:
            # try suffixes for sidecars too
            for suf in MASK_NAME_HINTS:
                if s + suf in cxy_map:
                    cxy = cxy_map[s + suf][0]
                    break
        if cxy:
            # find radius by same stem if present
            if s in rad_map:
                rad = rad_map[s][0]
            else:
                for suf in MASK_NAME_HINTS:
                    if s + suf in rad_map:
                        rad = rad_map[s + suf][0]
                        break
            pairs.append((ip, None, cxy, rad))
        else:
            missing += 1

    t2 = time.time()
    print(f"[index] pairs made: {len(pairs)} | images without label: {missing} | took {t2 - t1:.2f}s")

    # preview few
    for i, (a, b, c, r) in enumerate(pairs[:5]):
        print(f"[pair {i+1}] img: {a}")
        if b: print(f"          msk: {b}")
        if c: print(f"          cxy: {c}")
        if r: print(f"          rad: {r}")

    if not pairs:
        raise RuntimeError("No image+label pairs found under ./data/kaggle.")

    return pairs

def load_or_build_index(root: str, cache_path: str) -> List[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                pairs = pickle.load(f)
            print(f"[index] loaded cache: {cache_path} ({len(pairs)} pairs)")
            return pairs
        except Exception:
            print("[index] cache load failed, rebuilding...")

    # Try generic scan first; if it fails or finds 0, try CBIS-DDSM CSV mapping
    try:
        pairs = build_index_fast(root)
    except Exception as e:
        print(f"[index] fast scan failed: {e}")
        pairs = []
    if not pairs:
        print("[index] attempting CBIS-DDSM CSV-based index (train split) ...")
        pairs = build_index_cbis_csv(root)
    ensure_dir(os.path.dirname(cache_path))
    with open(cache_path, "wb") as f:
        pickle.dump(pairs, f)
    print(f"[index] saved cache: {cache_path}")
    return pairs

def build_index_cbis_csv(root: str) -> List[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    """
    Build image/mask pairs for CBIS-DDSM using provided CSVs that map DICOM SOPInstanceUIDs
    to JPEG paths and list the ROI mask DICOMs for each case. Returns list of (img, mask, None, None).
    """
    csv_dir = os.path.join(root, 'csv')
    di_path = os.path.join(csv_dir, 'dicom_info.csv')
    mass_csv = os.path.join(csv_dir, 'mass_case_description_train_set.csv')
    calc_csv = os.path.join(csv_dir, 'calc_case_description_train_set.csv')
    if not os.path.exists(di_path):
        raise RuntimeError("dicom_info.csv not found; cannot build CBIS-DDSM index.")

    # Build lookups from dicom_info.csv
    sop_to_jpeg: Dict[str, str] = {}
    series_to_jpegs: Dict[str, List[str]] = {}
    series_desc_to_jpegs: Dict[str, Dict[str, List[str]]] = {}
    with open(di_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            sop = row.get('SOPInstanceUID')
            img_path = row.get('image_path') or ''  # e.g., CBIS-DDSM/jpeg/<SeriesUID>/<name>.jpg
            series_uid = row.get('SeriesInstanceUID') or ''
            desc = (row.get('SeriesDescription') or '').lower()
            if not sop or not img_path:
                continue
            p = img_path.replace('\\', '/').split('/jpeg/', 1)
            rel = p[1] if len(p) > 1 else img_path.replace('CBIS-DDSM/', '')
            local = os.path.join(root, 'jpeg', rel).replace('\\', '/')
            sop_to_jpeg[sop] = local
            if series_uid:
                series_to_jpegs.setdefault(series_uid, []).append(local)
                bucket = 'other'
                if ('roi' in desc) and ('mask' in desc):
                    bucket = 'roi'
                elif 'cropped' in desc:
                    bucket = 'cropped'
                elif 'full' in desc:
                    bucket = 'full'
                series_desc_to_jpegs.setdefault(series_uid, {}).setdefault(bucket, []).append(local)

    def pairs_from_case_csv(path_csv: str) -> List[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
        pairs: List[Tuple[str, Optional[str], Optional[str], Optional[str]]] = []
        if not os.path.exists(path_csv):
            return pairs
        with open(path_csv, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                crop_dc = (row.get('cropped image file path') or '').replace('\\', '/')
                roi_dc = (row.get('ROI mask file path') or '').replace('\\', '/')
                # Extract SeriesInstanceUID and SOPInstanceUID (second from last token)
                def extract_series(p: str) -> Optional[str]:
                    parts = [q for q in p.split('/') if q]
                    return parts[-2] if len(parts) >= 2 else None
                def extract_sop(p: str) -> Optional[str]:
                    parts = [q for q in p.split('/') if q]
                    return parts[-2] if len(parts) >= 2 else None
                series_img = extract_series(crop_dc)
                series_roi = extract_series(roi_dc)
                sop_img = extract_sop(crop_dc)
                sop_roi = extract_sop(roi_dc)

                img_jpg = sop_to_jpeg.get(sop_img)
                msk_jpg = sop_to_jpeg.get(sop_roi)

                # Fallbacks if not found or identical
                if (not img_jpg or not os.path.exists(img_jpg)) and series_img:
                    cand = series_desc_to_jpegs.get(series_img, {})
                    img_jpg = (cand.get('cropped') or cand.get('full') or cand.get('other') or [None])[0]
                if (not msk_jpg or not os.path.exists(msk_jpg)) and series_roi:
                    candm = series_desc_to_jpegs.get(series_roi, {})
                    msk_jpg = (candm.get('roi') or candm.get('other') or [None])[0]

                if img_jpg and msk_jpg and os.path.exists(img_jpg) and os.path.exists(msk_jpg) and img_jpg != msk_jpg:
                    pairs.append((img_jpg, msk_jpg, None, None))
        return pairs

    pairs: List[Tuple[str, Optional[str], Optional[str], Optional[str]]] = []
    pairs += pairs_from_case_csv(mass_csv)
    pairs += pairs_from_case_csv(calc_csv)
    print(f"[index/CBIS] built {len(pairs)} pairs from CSVs (train)")
    if not pairs:
        raise RuntimeError("CBIS-DDSM CSV build produced 0 pairs. Check dataset structure.")
    return pairs

def build_index_cbis_csv_split(
    root: str,
    split: str,
    datasets: Optional[Tuple[str, ...]] = None,
    pathology_filter: Optional[str] = None,
) -> List[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    """Build pairs for a specific split: 'train' or 'test' using CBIS CSVs."""
    csv_dir = os.path.join(root, 'csv')
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if datasets is None:
        datasets = DATASETS
    ds = set(datasets)
    files: List[str] = []
    if split == "train":
        if "mass" in ds:
            files.append(os.path.join(csv_dir, 'mass_case_description_train_set.csv'))
        if "calc" in ds:
            files.append(os.path.join(csv_dir, 'calc_case_description_train_set.csv'))
    else:
        if "mass" in ds:
            files.append(os.path.join(csv_dir, 'mass_case_description_test_set.csv'))
        if "calc" in ds:
            files.append(os.path.join(csv_dir, 'calc_case_description_test_set.csv'))

    # Reuse dicom_info mapping logic
    di_path = os.path.join(csv_dir, 'dicom_info.csv')
    if not os.path.exists(di_path):
        raise RuntimeError("dicom_info.csv not found; cannot build CBIS-DDSM split index.")

    sop_to_jpeg: Dict[str, str] = {}
    series_desc_to_jpegs: Dict[str, Dict[str, List[str]]] = {}
    with open(di_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            sop = row.get('SOPInstanceUID')
            img_path = row.get('image_path') or ''
            series_uid = row.get('SeriesInstanceUID') or ''
            desc = (row.get('SeriesDescription') or '').lower()
            if not sop or not img_path:
                continue
            p = img_path.replace('\\', '/').split('/jpeg/', 1)
            rel = p[1] if len(p) > 1 else img_path.replace('CBIS-DDSM/', '')
            local = os.path.join(root, 'jpeg', rel).replace('\\', '/')
            sop_to_jpeg[sop] = local
            if series_uid:
                bucket = 'other'
                if ('roi' in desc) and ('mask' in desc):
                    bucket = 'roi'
                elif 'cropped' in desc:
                    bucket = 'cropped'
                elif 'full' in desc:
                    bucket = 'full'
                series_desc_to_jpegs.setdefault(series_uid, {}).setdefault(bucket, []).append(local)

    def extract_series(p: str) -> Optional[str]:
        parts = [q for q in p.replace('\\', '/').split('/') if q]
        return parts[-2] if len(parts) >= 2 else None

    def extract_sop(p: str) -> Optional[str]:
        parts = [q for q in p.replace('\\', '/').split('/') if q]
        return parts[-2] if len(parts) >= 2 else None

    pairs: List[Tuple[str, Optional[str], Optional[str], Optional[str]]] = []
    for path_csv in files:
        if not os.path.exists(path_csv):
            continue
        with open(path_csv, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                if pathology_filter:
                    p = (row.get('pathology') or row.get('Pathology') or '').strip().upper()
                    if p != pathology_filter.upper():
                        continue
                crop_dc = (row.get('cropped image file path') or '')
                roi_dc = (row.get('ROI mask file path') or '')
                series_img = extract_series(crop_dc)
                series_roi = extract_series(roi_dc)
                sop_img = extract_sop(crop_dc)
                sop_roi = extract_sop(roi_dc)

                img_jpg = sop_to_jpeg.get(sop_img)
                msk_jpg = sop_to_jpeg.get(sop_roi)

                if (not img_jpg or not os.path.exists(img_jpg)) and series_img:
                    cand = series_desc_to_jpegs.get(series_img, {})
                    img_jpg = (cand.get('cropped') or cand.get('full') or cand.get('other') or [None])[0]
                if (not msk_jpg or not os.path.exists(msk_jpg)) and series_roi:
                    candm = series_desc_to_jpegs.get(series_roi, {})
                    msk_jpg = (candm.get('roi') or candm.get('other') or [None])[0]

                if img_jpg and msk_jpg and os.path.exists(img_jpg) and os.path.exists(msk_jpg) and img_jpg != msk_jpg:
                    pairs.append((img_jpg, msk_jpg, None, None))

    print(f"[index/CBIS:{split}] built {len(pairs)} pairs from CSVs")
    if not pairs:
        raise RuntimeError(f"CBIS-DDSM CSV build for split '{split}' produced 0 pairs.")
    return pairs

# =========================
# Dataset
# =========================
class SegDataset(Dataset):
    def __init__(self, triples: List[Tuple[str, Optional[str], Optional[str], Optional[str]]], img_size: int):
        self.triples = triples
        self.img_size = img_size

    def __len__(self):
        return len(self.triples)

    def _synth_mask_from_cxy(self, w: int, h: int, cxy_path: str, rad_path: Optional[str]) -> Image.Image:
        nums = parse_numbers_from_file(cxy_path)
        cx, cy = (nums[0], nums[1]) if len(nums) >= 2 else (None, None)
        r = DEFAULT_RAD_PX
        if rad_path:
            rnums = parse_numbers_from_file(rad_path)
            if len(rnums) >= 1 and rnums[0] > 0:
                r = int(rnums[0])
        mask = np.zeros((h, w), dtype=np.uint8)
        if cx is not None and cy is not None:
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(r), 255, thickness=-1)
        return Image.fromarray(mask, mode="L")

    def __getitem__(self, idx: int):
        img_path, mask_img_path, cxy_path, rad_path = self.triples[idx]

        img = Image.open(img_path).convert("L")
        w, h = img.size

        if mask_img_path:
            msk = Image.open(mask_img_path).convert("L")
        else:
            msk = self._synth_mask_from_cxy(w, h, cxy_path, rad_path)

        img = resize_img(img, self.img_size)
        msk = resize_mask(msk, self.img_size)

        # light flips
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            msk = msk.transpose(Image.FLIP_TOP_BOTTOM)

        img_t = pil_to_tensor_image(img)
        msk_t = pil_to_tensor_mask(msk)
        return img_t, msk_t, img_path

# =========================
# Model
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.outc(d1)

# =========================
# TransUNet (Transformer bottleneck)
# =========================

def _pos_get_1d(pos: torch.Tensor, dim_half: int, device: torch.device) -> torch.Tensor:
    omega = torch.arange(dim_half, dtype=torch.float32, device=device)
    omega = 1.0 / torch.pow(10000.0, (omega / dim_half))
    out = pos.reshape(-1, 1) * omega.reshape(1, -1)
    sin = torch.sin(out)
    cos = torch.cos(out)
    return torch.cat([sin, cos], dim=1)

def get_2d_sincos_pos_embed(embed_dim: int, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Build 2D sin-cos positional embeddings (h*w, embed_dim) on a specific device."""
    assert embed_dim % 2 == 0, "embed_dim must be even"
    if device is None:
        device = torch.device("cpu")
    grid_y = torch.arange(h, dtype=torch.float32, device=device)
    grid_x = torch.arange(w, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(grid_y, grid_x)  # default 'ij'

    dim_h = embed_dim // 2
    dim_h_half = dim_h // 2
    dim_w_half = dim_h // 2
    emb_y = _pos_get_1d(gy, dim_h_half, device)  # (h*w, dim_h)
    emb_x = _pos_get_1d(gx, dim_w_half, device)  # (h*w, dim_h)
    pos = torch.cat([emb_y, emb_x], dim=1)  # (h*w, embed_dim)
    return pos


class TransformerBottleneck(nn.Module):
    def __init__(self, channels: int, nhead: int = 8, depth: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        # Use PyTorch TransformerEncoder with d_model=channels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=nhead, dim_feedforward=int(channels * mlp_ratio), dropout=dropout, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        S = H * W
        x_seq = x.view(B, C, S).permute(2, 0, 1)  # (S, B, C)
        pos = get_2d_sincos_pos_embed(C, H, W, device=x_seq.device)  # (S, C)
        x_seq = x_seq + pos.unsqueeze(1)  # (S, B, C)
        x_enc = self.encoder(x_seq)  # (S, B, C)
        x_out = x_enc.permute(1, 2, 0).view(B, C, H, W)
        return x_out


class TransUNetSmall(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32, nhead: int = 8, depth: int = 4):
        super().__init__()
        # Encoder (same as UNetSmall)
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Transformer bottleneck on 1/16 res with C=base*16 (keeps channels constant through transformer)
        self.pre_bottleneck = DoubleConv(base * 8, base * 16)
        self.trans = TransformerBottleneck(base * 16, nhead=nhead, depth=depth)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.pre_bottleneck(self.pool4(e4))
        b = self.trans(b)
        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.outc(d1)

# =========================
# Export helpers
# =========================
def build_model() -> nn.Module:
    return TransUNetSmall(in_ch=1, out_ch=1, base=32, nhead=8, depth=4)

def load_model(weights: Optional[str] = None, device: Optional[str] = None) -> nn.Module:
    device = device or DEVICE
    m = build_model().to(device)
    w = weights or WEIGHTS_PATH
    if not os.path.exists(w):
        raise FileNotFoundError(f"Weights not found at {w}. Train or provide a valid path.")
    state = torch.load(w, map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m

def export_torchscript(weights: Optional[str] = None, out_path: Optional[str] = None, img_size: int = IMG_SIZE) -> str:
    ensure_dir(OUT_DIR)
    out_path = out_path or EXPORT_TS_PATH
    model = load_model(weights, DEVICE)
    # Script the model to keep device-agnostic constants and support dynamic devices
    scripted = torch.jit.script(model)
    scripted.save(out_path)
    print(f"Exported TorchScript -> {out_path}")
    return out_path

    

# =========================
# Loss / metric
# =========================
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        num = 2.0 * (probs * targets).sum(dim=(2, 3)) + smooth
        den = (probs + targets).sum(dim=(2, 3)) + smooth
        dice = 1.0 - (num / den)
        return bce + dice.mean()

def dice_score(probs, targets) -> float:
    probs_bin = (probs > 0.5).float()
    num = 2.0 * (probs_bin * targets).sum(dim=(2, 3))
    den = (probs_bin + targets).sum(dim=(2, 3)) + 1e-7
    return (num / den).mean().item()

# =========================
# Split helpers (fixed once)
# =========================
def save_split(triples, path_csv):
    random.seed(SEED)
    idx = list(range(len(triples)))
    random.shuffle(idx)
    n_val = max(1, int(len(triples) * VAL_SPLIT))
    val_idx = set(idx[:n_val])
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split","img","mask","cxy","rad"])
        for i,(a,b,c,r) in enumerate(triples):
            split = "val" if i in val_idx else "train"
            w.writerow([split, a, b or "", c or "", r or ""])

def load_split(path_csv):
    train, val = [], []
    with open(path_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            item = (row["img"], row["mask"] or None, row["cxy"] or None, row["rad"] or None)
            if row["split"] == "train":
                train.append(item)
            else:
                val.append(item)
    return train, val

# =========================
# Train / Val
# =========================
def train_loop(datasets: Optional[Tuple[str, ...]] = None, pathology_filter: Optional[str] = None):
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    # Build splits
    t0 = time.time()
    if USE_CSV_SPLITS:
        ds_use = datasets if datasets is not None else DATASETS
        pf_use = pathology_filter if pathology_filter is not None else PATHOLOGY_FILTER
        train_triples = build_index_cbis_csv_split(ROOT, 'train', datasets=ds_use, pathology_filter=pf_use)
        val_triples = build_index_cbis_csv_split(ROOT, 'test', datasets=ds_use, pathology_filter=pf_use)
    else:
        triples = load_or_build_index(ROOT, INDEX_PKL)
        if not os.path.exists(SPLIT_CSV):
            print("[split] creating split.csv ...")
            save_split(triples, SPLIT_CSV)
        train_triples, val_triples = load_split(SPLIT_CSV)
    t1 = time.time()
    print(f"[split] train: {len(train_triples)} | val: {len(val_triples)} | took {t1 - t0:.2f}s")

    train_ds = SegDataset(train_triples, IMG_SIZE)
    val_ds = SegDataset(val_triples, IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS_TRAIN, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS_VAL, pin_memory=True)

    model = TransUNetSmall(in_ch=1, out_ch=1, base=32, nhead=8, depth=4)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, verbose=True)
    loss_fn = BCEDiceLoss()

    # CSV log header
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","val_dice"])

    best_val = math.inf

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
        tr_loss /= max(1, len(train_loader.dataset))

        # Val
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(logits)
                val_dice += dice_score(probs, masks) * imgs.size(0)
        val_loss /= max(1, len(val_loader.dataset))
        val_dice /= max(1, len(val_loader.dataset))

        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")
        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, val_loss, val_dice])

        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  saved best -> {WEIGHTS_PATH}")

    print(f"Training done. Log: {LOG_CSV}")

# =========================
# Inference + circle overlay
# =========================
def min_enclosing_circle(mask_bin: np.ndarray):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    (x, y), r = cv2.minEnclosingCircle(contours[0])
    return int(x), int(y), int(r)

def infer_and_draw(sample_count: int = 12, thr: float = 0.25, min_area: int = 64):
    ensure_dir(OUT_DIR)
    overlay_dir = os.path.join(OUT_DIR, "overlays")
    if os.path.exists(overlay_dir):
        shutil.rmtree(overlay_dir)
    ensure_dir(overlay_dir)

    model = TransUNetSmall(in_ch=1, out_ch=1, base=32, nhead=8, depth=4).to(DEVICE)
    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError("Weights not found. Train first.")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    triples = load_or_build_index(ROOT, INDEX_PKL)
    random.seed(SEED)
    random.shuffle(triples)
    triples = triples[:sample_count]

    for img_path, _, _, _ in triples:
        img = Image.open(img_path).convert("L")
        img_r = resize_img(img, IMG_SIZE)
        ten = pil_to_tensor_image(img_r).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(ten)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Save heatmap overlay for debugging
        prob_u8 = (np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)

        # Threshold + morphology
        thr_u8 = int(max(0, min(1, thr)) * 255)
        _, mask_bin = cv2.threshold(prob_u8, thr_u8, 255, cv2.THRESH_BINARY)
        mask_bin = cv2.medianBlur(mask_bin, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Drop tiny speckles
        cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_clean = np.zeros_like(mask_bin)
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(mask_clean, [c], -1, 255, -1)
        mask_bin = mask_clean

        vis = cv2.cvtColor(np.array(img_r), cv2.COLOR_GRAY2BGR)
        circle = min_enclosing_circle(mask_bin)
        if circle:
            cx, cy, r = circle
            cv2.circle(vis, (cx, cy), r, (0, 0, 255), 2)
        else:
            # Fallback: draw a small circle at the probability peak so you can see model focus
            peak = np.unravel_index(np.argmax(prob_u8), prob_u8.shape)
            cy, cx = int(peak[0]), int(peak[1])
            cv2.circle(vis, (cx, cy), 12, (0, 255, 255), 2)

        blend = cv2.addWeighted(vis, 0.7, cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR), 0.3, 0)
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(overlay_dir, f"{base}_overlay.png")
        heat_path = os.path.join(overlay_dir, f"{base}_heat.png")
        cv2.imwrite(out_path, blend)
        cv2.imwrite(heat_path, cv2.addWeighted(vis, 0.6, heat, 0.4, 0))
        print(f"Saved {out_path}")

    print(f"Overlays saved to {overlay_dir}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    ensure_dir(OUT_DIR)

    parser = argparse.ArgumentParser(description="Train/Export TransUNet on CBIS-DDSM")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--infer", action="store_true", help="Run overlay inference samples")
    parser.add_argument("--samples", type=int, default=12, help="Number of overlays to generate with --infer")
    parser.add_argument("--export-torchscript", action="store_true", help="Export TorchScript model")
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH, help="Path to weights .pt (for export/infer)")
    parser.add_argument("--ts-out", type=str, default=EXPORT_TS_PATH, help="TorchScript output path")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Export input size (HxW)")
    parser.add_argument("--datasets", type=str, choices=["mass", "calc", "both"], default="both", help="Which CBIS subsets to use")
    parser.add_argument("--pathology", type=str, choices=["ANY", "MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"], default="ANY", help="Filter by pathology label")
    args = parser.parse_args()

    any_action = args.train or args.infer or args.export_torchscript

    # Apply dataset selection and pathology filter (pass to train_loop)
    ds_cli = ("mass", "calc") if args.datasets == "both" else (args.datasets,)
    pf_cli = None if args.pathology == "ANY" else args.pathology

    if not any_action:
        # Default behavior: train, export TorchScript, then quick infer sample set
        train_loop(datasets=ds_cli, pathology_filter=pf_cli)
        export_torchscript(weights=WEIGHTS_PATH, out_path=EXPORT_TS_PATH, img_size=IMG_SIZE)
        infer_and_draw(sample_count=12)
    else:
        if args.train:
            train_loop(datasets=ds_cli, pathology_filter=pf_cli)
            # Export by default after training completes
            export_torchscript(weights=WEIGHTS_PATH, out_path=EXPORT_TS_PATH, img_size=IMG_SIZE)
        if args.export_torchscript:
            export_torchscript(weights=args.weights, out_path=args.ts_out, img_size=args.img_size)
        if args.infer:
            infer_and_draw(sample_count=args.samples)
