import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Optional  # ← 追加


# ===== パス設定 =====
target_folder   = Path("pix2pix/targets")     # 色塗り画像の保存先
combined_folder = Path("pix2pix/combined")    # 元画像＋色塗りの保存先
xml_folder          = Path("annotation")      # PascalVOC形式XML
input_image_folder  = Path("captured_frames") # 画像

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# ===== ユーティリティ =====
def find_image(stem: str) -> Optional[Path]:
    """XMLのstemに一致する画像を拡張子違いも含めて探す"""
    for ext in IMG_EXTS:
        p = input_image_folder / f"{stem}{ext}"
        if p.exists():
            return p
        p2 = input_image_folder / f"{stem}{ext.upper()}"
        if p2.exists():
            return p2
    return None

def clip_box(box, w, h):
    """(xmin, ymin, xmax, ymax) を画像サイズ内にクリップ。無効なら None"""
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

# ===== メイン =====
def main():
    target_folder.mkdir(parents=True, exist_ok=True)
    combined_folder.mkdir(parents=True, exist_ok=True)

    xml_list = sorted([p for p in xml_folder.glob("*.xml")])
    if not xml_list:
        print("XMLが見つかりませんでした。処理終了。")
        return

    ok = skip = err = 0

    for xml_path in xml_list:
        stem = xml_path.stem
        img_path = find_image(stem)
        if img_path is None:
            print(f"[SKIP] 画像なし: {stem}")
            skip += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            mask = Image.new("RGB", (w, h), (0, 0, 0))
            draw = ImageDraw.Draw(mask)

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # PascalVOC: objectタグのみ走査
            for obj in root.findall("object"):
                name_tag = obj.find("name")
                bb = obj.find("bndbox")
                if name_tag is None or bb is None:
                    continue

                name = (name_tag.text or "").strip()
                try:
                    xmin = int(float(bb.findtext("xmin")))
                    ymin = int(float(bb.findtext("ymin")))
                    xmax = int(float(bb.findtext("xmax")))
                    ymax = int(float(bb.findtext("ymax")))
                except Exception:
                    continue

                # ここで必要なラベルだけ塗る（例: Kirby）
                if name == "Kirby":
                    box = clip_box((xmin, ymin, xmax, ymax), w, h)
                    if box:
                        draw.rectangle(box, fill=(255, 0, 255))  # マゼンタ

                # 他ラベルを追加する場合は以下のように
                # elif name == "class":
                #     box = clip_box((xmin, ymin, xmax, ymax), w, h)
                #     if box: draw.rectangle(box, fill=(255, 0, 0))

            # 保存
            out_mask = target_folder / f"{stem}.jpg"
            mask.save(out_mask, quality=95)

            combined = Image.new("RGB", (w * 2, h))
            combined.paste(image, (0, 0))
            combined.paste(mask, (w, 0))
            out_comb = combined_folder / f"{stem}.jpg"
            combined.save(out_comb, quality=95)

            print(f"[OK] {stem}")
            ok += 1

        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            err += 1

    print("\n=== 結果 ===")
    print(f"成功: {ok} / スキップ: {skip} / エラー: {err}")

if __name__ == "__main__":
    main()