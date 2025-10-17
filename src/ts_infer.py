import os
import glob
import argparse
import numpy as np
from PIL import Image
import cv2
import torch

DEFAULT_TS = os.path.join("model", "kaggle", "out_seg", "transunet.ts")
DEFAULT_IMG_ROOT = os.path.join("data", "kaggle", "jpeg")
OUT_DIR = os.path.join("model", "kaggle", "out_seg")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_first_image(root: str) -> str:
    pats = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp"]
    for pat in pats:
        hits = glob.glob(os.path.join(root, pat), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No images found under {root}")


def preprocess(pil: Image.Image, size: int) -> torch.Tensor:
    pil = pil.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return ten


def colorize_heatmap(gray: np.ndarray) -> np.ndarray:
    gray_u8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)


def min_enclosing_circle(mask_bin: np.ndarray):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    (x, y), r = cv2.minEnclosingCircle(contours[0])
    return int(x), int(y), int(r)


def compute_size_metrics(mask_bin: np.ndarray, mm_per_pixel: float | None = None):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area_px": 0,
            "radius_px": 0.0,
            "diameter_px": 0.0,
            "area_mm2": None,
            "radius_mm": None,
            "diameter_mm": None,
        }
    c = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(c))
    (cx, cy), r = cv2.minEnclosingCircle(c)
    radius_px = float(r)
    diameter_px = 2.0 * radius_px
    if mm_per_pixel is not None and mm_per_pixel > 0:
        radius_mm = radius_px * mm_per_pixel
        diameter_mm = diameter_px * mm_per_pixel
        area_mm2 = area_px * (mm_per_pixel ** 2)
    else:
        radius_mm = diameter_mm = area_mm2 = None
    return {
        "area_px": area_px,
        "radius_px": radius_px,
        "diameter_px": diameter_px,
        "area_mm2": area_mm2,
        "radius_mm": radius_mm,
        "diameter_mm": diameter_mm,
    }


def save_outputs(pil_resized: Image.Image, prob: np.ndarray, out_prefix: str, thr: float):
    prob_u8 = (prob * 255).astype(np.uint8)
    _, mask_bin = cv2.threshold(prob_u8, 127, 255, cv2.THRESH_BINARY)
    vis = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_GRAY2BGR)
    # Binary overlay (threshold configurable)
    thr_u8 = int(max(0, min(1, thr)) * 255)
    _, mask_thr = cv2.threshold(prob_u8, thr_u8, 255, cv2.THRESH_BINARY)
    overlay = cv2.addWeighted(vis, 0.75, cv2.cvtColor(mask_thr, cv2.COLOR_GRAY2BGR), 0.25, 0)
    circ = min_enclosing_circle(mask_thr)
    if circ:
        cx, cy, r = circ
        cv2.circle(overlay, (cx, cy), r, (0, 0, 255), 2)
    # Heatmap overlay from probabilities
    heat = colorize_heatmap(prob)
    heat_overlay = cv2.addWeighted(vis, 0.6, heat, 0.4, 0)
    ensure_dir(OUT_DIR)
    mask_path = os.path.join(OUT_DIR, f"{out_prefix}_ts_mask.png")
    overlay_path = os.path.join(OUT_DIR, f"{out_prefix}_ts_overlay.png")
    heat_path = os.path.join(OUT_DIR, f"{out_prefix}_ts_heat.png")
    cv2.imwrite(mask_path, mask_thr)
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(heat_path, heat_overlay)
    print(f"Saved {mask_path}")
    print(f"Saved {overlay_path}")
    print(f"Saved {heat_path}")
    if circ:
        print(f"Predicted circle: cx={cx}, cy={cy}, r={r}")


def main():
    parser = argparse.ArgumentParser(description="TorchScript inference smoke test")
    parser.add_argument("--ts", type=str, default=DEFAULT_TS, help="Path to transunet.ts")
    parser.add_argument("--img", type=str, default=None, help="Path to an input image (grayscale is fine)")
    parser.add_argument("--img-size", type=int, default=512, help="Resize to this size before inference")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--thr", type=float, default=0.35, help="Probability threshold for binary mask [0-1]")
    parser.add_argument("--mm-per-pixel", type=float, default=None, help="Pixel spacing in mm/pixel for size in mm")
    args = parser.parse_args()

    if not os.path.exists(args.ts):
        raise FileNotFoundError(f"TorchScript not found at {args.ts}. Run training export first.")

    if args.img is None:
        img_path = find_first_image(DEFAULT_IMG_ROOT)
    else:
        img_path = args.img
    print(f"Using image: {img_path}")

    model = torch.jit.load(args.ts, map_location=args.device).eval()

    pil = Image.open(img_path).convert("L")
    pil_r = pil.resize((args.img_size, args.img_size), Image.BILINEAR)
    x = preprocess(pil, args.img_size).to(args.device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    base = os.path.splitext(os.path.basename(img_path))[0]
    # Save overlays
    save_outputs(pil_r, prob, out_prefix=base, thr=args.thr)

    # Compute and print size metrics
    prob_u8 = (np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
    thr_u8 = int(max(0, min(1, args.thr)) * 255)
    _, mask_bin = cv2.threshold(prob_u8, thr_u8, 255, cv2.THRESH_BINARY)
    mask_bin = cv2.medianBlur(mask_bin, 5)
    metrics = compute_size_metrics(mask_bin, mm_per_pixel=args.mm_per_pixel)
    print("Size (pixels): area=%.1f, radius=%.2f, diameter=%.2f" % (
        metrics["area_px"], metrics["radius_px"], metrics["diameter_px"]
    ))
    if metrics["area_mm2"] is not None:
        print("Size (mm): area=%.2f mm^2, radius=%.2f mm, diameter=%.2f mm" % (
            metrics["area_mm2"], metrics["radius_mm"], metrics["diameter_mm"]
        ))
    # Persist metrics alongside images
    out_txt = os.path.join(OUT_DIR, f"{base}_ts_metrics.txt")
    with open(out_txt, "w") as f:
        f.write(str(metrics))
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
