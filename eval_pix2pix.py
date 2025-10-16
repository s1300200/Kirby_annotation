# eval_pix2pix.py
# 評価対象:
#   - 生成:  pix2pix/infer_outputs/*.png|jpg
#   - 正解:  pix2pix/combined/<同名> を右半分で切り出して参照 (TARGET_DIR=None の場合)
# 追加機能:
#   - 色ラベルのIoU (任意): COLORS に辞書を設定すると色ごとにIoU算出

import os, csv
from typing import Tuple, Dict
import numpy as np
from PIL import Image

INFER_DIR   = "pix2pix/infer_outputs"   # 生成画像
COMBINED_DIR= "pix2pix/combined"        # combined（右半分がGT）
TARGET_DIR  = "pix2pix/targets"                   # 別にGTフォルダがあるならパスを入れる。なければ None で combined を使う
IMG_SIZE    = (256, 256)                # 学習解像度に合わせる

# ----（任意）色→クラス名の定義。空ならIoUはスキップ ----
# 例: Kirby領域がマゼンタなら:
COLORS: Dict[str, Tuple[int,int,int]] = {
    "Kirby": (255, 0, 255),
    # 他クラスがあるなら追加:
    # "hiragana": (0, 0, 255),
    # "region":   (0, 255, 0),
    # "class":    (255, 0, 0),
}
COLOR_TOL = 10  # 色の許容幅（±）

def load_img(path: str, size: Tuple[int,int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size, Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 255.0

def crop_target_from_combined(path: str, size: Tuple[int,int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    w2 = w // 2
    right = img.crop((w2, 0, w, h)).resize(size, Image.BICUBIC)
    return np.array(right, dtype=np.float32) / 255.0

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m == 0:
        return 99.0
    return 10.0 * np.log10(1.0 / m)

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def color_mask(img: np.ndarray, rgb: Tuple[int,int,int], tol: int = 10) -> np.ndarray:
    # img: [H,W,3] 0..1, rgb: 0..255
    target = np.array(rgb, dtype=np.float32) / 255.0
    diff = np.abs(img - target[None, None, :])
    # 3chすべて許容範囲内を“そのクラス”とみなす
    m = (diff <= (tol/255.0)).all(axis=2)
    return m.astype(np.uint8)

def iou(pred_m: np.ndarray, gt_m: np.ndarray) -> float:
    inter = np.logical_and(pred_m==1, gt_m==1).sum()
    union = np.logical_or (pred_m==1, gt_m==1).sum()
    return float(inter / union) if union > 0 else 1.0  # union=0の時は完全一致扱い

def main():
    files = [f for f in os.listdir(INFER_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not files:
        print(f"No images found in {INFER_DIR}")
        return

    out_csv = os.path.join(INFER_DIR, "metrics.csv")
    rows = []
    agg = {"mae":[], "mse":[], "psnr":[]}
    agg_iou = {k: [] for k in COLORS.keys()}

    for f in sorted(files):
        infer_path = os.path.join(INFER_DIR, f)
        pred = load_img(infer_path, IMG_SIZE)

        # GTの取得
        if TARGET_DIR:
            gt_path = os.path.join(TARGET_DIR, f)
            if not os.path.exists(gt_path):
                print(f"[SKIP] GT not found for {f}")
                continue
            gt = load_img(gt_path, IMG_SIZE)
        else:
            comb_path = os.path.join(COMBINED_DIR, f)
            if not os.path.exists(comb_path):
                print(f"[SKIP] combined not found for {f}")
                continue
            gt = crop_target_from_combined(comb_path, IMG_SIZE)

        # メトリクス
        m_mae  = mae(pred, gt)
        m_mse  = mse(pred, gt)
        m_psnr = psnr(pred, gt)

        row = {"file": f, "mae": m_mae, "mse": m_mse, "psnr": m_psnr}

        # 色IoU（任意）
        if COLORS:
            for name, rgb in COLORS.items():
                pm = color_mask((pred*255).astype(np.uint8)/255.0, rgb, COLOR_TOL)
                gm = color_mask((gt*255).astype(np.uint8)/255.0,   rgb, COLOR_TOL)
                j = iou(pm, gm)
                row[f"iou_{name}"] = j
                agg_iou[name].append(j)

        rows.append(row)
        agg["mae"].append(m_mae); agg["mse"].append(m_mse); agg["psnr"].append(m_psnr)

    # CSV保存
    keys = list(rows[0].keys()) if rows else ["file","mae","mse","psnr"]
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow(r)

    # 集計
    def avg(x): return float(np.mean(x)) if x else float("nan")
    print("\n=== Summary ===")
    print(f"count: {len(rows)}")
    print(f"MAE : {avg(agg['mae']):.4f}")
    print(f"MSE : {avg(agg['mse']):.6f}")
    print(f"PSNR: {avg(agg['psnr']):.2f} dB")
    if COLORS:
        for name, vals in agg_iou.items():
            print(f"IoU[{name}]: {avg(vals):.4f}")
    print(f"\nMetrics CSV -> {out_csv}")

if __name__ == "__main__":
    main()

#出力結果
'''
[SKIP] GT not found for frame_00627.jpg
[SKIP] GT not found for frame_00627_反転.jpg
[SKIP] GT not found for frame_06642.jpg
[SKIP] GT not found for frame_06642_反転.jpg
[SKIP] GT not found for frame_08131_反転.jpg
[SKIP] GT not found for frame_08937.jpg
[SKIP] GT not found for frame_08937_反転.jpg
[SKIP] GT not found for frame_09283.jpg
[SKIP] GT not found for frame_09283_反転.jpg
[SKIP] GT not found for frame_09516.jpg
[SKIP] GT not found for frame_09516_反転.jpg

=== Summary ===
count: 141
MAE : 0.0179
MSE : 0.015870
PSNR: 21.94 dB
IoU[Kirby]: 0.3403

'''