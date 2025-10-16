from PIL import Image
from pathlib import Path

# === 設定 ===
input_dir = Path("captured_frames")   # 元画像フォルダ
output_dir = Path("captured_frames2")  # 保存先フォルダ
output_dir.mkdir(parents=True, exist_ok=True)

# jpgを対象にする
images = list(input_dir.glob("*.jpg"))

print(f"🔍 {len(images)} 枚の画像を処理します")

for img_path in images:
    try:
        img = Image.open(img_path)
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)  # ← 左右反転
        save_path = output_dir / img_path + "_2" + .name
        flipped.save(save_path, quality=95)
        print(f"✅ {img_path.name} → {save_path.name}")
    except Exception as e:
        print(f"⚠️ {img_path.name} エラー: {e}")

print("\n🎉 完了！反転画像を以下に保存しました:")
print(output_dir)