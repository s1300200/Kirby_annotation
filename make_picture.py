import cv2
import os

# 動画ファイルのパス
video_path = "kirby_discovery.MP4"  # 任意の動画ファイル名に変更
output_dir = "captured_frames"

# 出力フォルダを作成
os.makedirs(output_dir, exist_ok=True)

# 動画を読み込み
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("動画ファイルを開けませんでした。")
    exit()

frame_count = 0
saved_count = 0

print("▶ 再生中：スペースキーでフレーム保存、'q'で終了")

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 動画の終わりです。")
        break

    frame_count += 1
    cv2.imshow("Video", frame)

    # キー入力待ち（1ミリ秒）
    key = cv2.waitKey(1) & 0xFF

    # スペースキーでフレーム保存
    if key == ord(' '):
        filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
        print(f"💾 保存しました: {filename}")

    # qキーで終了
    elif key == ord('q'):
        print("🛑 終了します。")
        break

# 後処理
cap.release()
cv2.destroyAllWindows()

print(f"✅ 保存した画像枚数: {saved_count}")
