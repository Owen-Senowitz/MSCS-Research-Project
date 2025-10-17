import os
import argparse
import random
import cv2
import numpy as np
from PIL import Image

from src.train_kaggle_dataset import (
    build_index_cbis_csv,
    resize_img,
    resize_mask,
)

OUT_DIR = os.path.join("model", "kaggle", "pair_inspect")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Visual check of CBIS-DDSM image/mask pairing")
    parser.add_argument("--root", type=str, default="./data/kaggle", help="Dataset root")
    parser.add_argument("--num", type=int, default=12, help="Number of samples to render")
    parser.add_argument("--size", type=int, default=512, help="Resize side")
    args = parser.parse_args()

    pairs = build_index_cbis_csv(args.root)
    random.seed(42)
    random.shuffle(pairs)
    ensure_dir(OUT_DIR)

    count = 0
    for img_path, msk_path, _, _ in pairs:
        try:
            img = Image.open(img_path).convert("L")
            msk = Image.open(msk_path).convert("L")
            img_r = resize_img(img, args.size)
            msk_r = resize_mask(msk, args.size)
            vis = cv2.cvtColor(np.array(img_r), cv2.COLOR_GRAY2BGR)
            msk_u8 = (np.array(msk_r) > 127).astype(np.uint8) * 255
            overlay = cv2.addWeighted(vis, 0.7, cv2.cvtColor(msk_u8, cv2.COLOR_GRAY2BGR), 0.3, 0)
            base = os.path.splitext(os.path.basename(img_path))[0]
            outp = os.path.join(OUT_DIR, f"{base}_pair_overlay.png")
            cv2.imwrite(outp, overlay)
            print("Saved", outp)
            count += 1
            if count >= args.num:
                break
        except Exception as e:
            print("skip", img_path, msk_path, "reason:", e)


if __name__ == "__main__":
    main()

