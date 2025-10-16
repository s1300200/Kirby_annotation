# inference_pix2pix.py
import os
from typing import Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

CKPT_PATH = "pix2pix_generator.pth"
INPUT_DIR = "captured_frames"   # 入力画像をここに置く（RGB）
OUTPUT_DIR = "pix2pix/infer_outputs"
IMG_SIZE = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def resize_pil(img: Image.Image, size: Tuple[int,int]) -> Image.Image:
    return img.resize(size, Image.BICUBIC)

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_dropout=False, norm='in'):
        super().__init__()
        if down:
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if norm != 'none':
                layers.append(nn.InstanceNorm2d(out_c, affine=True, track_running_stats=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.block = nn.Sequential(*layers)
        else:
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)]
            if norm != 'none':
                layers.append(nn.InstanceNorm2d(out_c, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetBlock(3,    64,  down=True,  norm='none')
        self.down2 = UNetBlock(64,   128, down=True,  norm='in')
        self.down3 = UNetBlock(128,  256, down=True,  norm='in')
        self.down4 = UNetBlock(256,  512, down=True,  norm='in')
        self.down5 = UNetBlock(512,  512, down=True,  norm='in')
        self.down6 = UNetBlock(512,  512, down=True,  norm='in')
        self.down7 = UNetBlock(512,  512, down=True,  norm='in')
        self.down8 = UNetBlock(512,  512, down=True,  norm='none')
        self.bottom = UNetBlock(512, 512, down=False, norm='in')
        self.up7 = UNetBlock(1024, 512, down=False, use_dropout=True, norm='in')
        self.up6 = UNetBlock(1024, 512, down=False, use_dropout=True, norm='in')
        self.up5 = UNetBlock(1024, 512, down=False, use_dropout=True, norm='in')
        self.up4 = UNetBlock(1024, 256, down=False, norm='in')
        self.up3 = UNetBlock(512,  128, down=False, norm='in')
        self.up2 = UNetBlock(256,  64,  down=False, norm='in')
        self.final = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2); d4 = self.down4(d3)
        d5 = self.down5(d4); d6 = self.down6(d5); d7 = self.down7(d6); d8 = self.down8(d7)
        b  = self.bottom(d8)
        u7 = self.up7(torch.cat([b,  d7], 1))
        u6 = self.up6(torch.cat([u7, d6], 1))
        u5 = self.up5(torch.cat([u6, d5], 1))
        u4 = self.up4(torch.cat([u5, d4], 1))
        u3 = self.up3(torch.cat([u4, d3], 1))
        u2 = self.up2(torch.cat([u3, d2], 1))
        return self.final(torch.cat([u2, d1], 1))

def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x*0.5 + 0.5).clamp(0,1)

def main():
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    G = GeneratorUNet().to(device)
    G.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    G.eval()

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not files:
        print(f"No images in {INPUT_DIR}")
        return

    for f in files:
        p = os.path.join(INPUT_DIR, f)
        img = Image.open(p).convert("RGB")
        img = pil_to_tensor(resize_pil(img, IMG_SIZE)) * 2 - 1
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            y = G(x)
        y_img = denorm(y[0]).permute(1,2,0).cpu().numpy()
        out = Image.fromarray((y_img*255).astype(np.uint8))
        out.save(os.path.join(OUTPUT_DIR, f))
        print("Saved:", f)

if __name__ == "__main__":
    main()

#出力結果
"""
python3 inference_pix2pix.py
Device: mps
Saved: frame_00501.jpg
Saved: frame_05774_反転.jpg
Saved: frame_04565.jpg
Saved: frame_03626.jpg
Saved: frame_03263_反転.jpg
Saved: frame_12295_反転.jpg
Saved: frame_01714_反転.jpg
Saved: frame_03543_反転.jpg
Saved: frame_00266.jpg
Saved: frame_09283.jpg
Saved: frame_02402_反転.jpg
Saved: frame_04229.jpg
Saved: frame_08167.jpg
Saved: frame_04417.jpg
Saved: frame_01553.jpg
Saved: frame_09341_反転.jpg
Saved: frame_07614_反転.jpg
Saved: frame_04827.jpg
Saved: frame_06444_反転.jpg
Saved: frame_03543.jpg
Saved: frame_05126.jpg
Saved: frame_05456.jpg
Saved: frame_08498_反転.jpg
Saved: frame_02875_反転.jpg
Saved: frame_00537_反転.jpg
Saved: frame_00266_反転.jpg
Saved: frame_01418.jpg
Saved: frame_01949.jpg
Saved: frame_07861.jpg
Saved: frame_02058.jpg
Saved: frame_00501_反転.jpg
Saved: frame_12771_反転.jpg
Saved: frame_08355_反転.jpg
Saved: frame_00984.jpg
Saved: frame_08937.jpg
Saved: frame_03658_反転.jpg
Saved: frame_09341.jpg
Saved: frame_00832.jpg
Saved: frame_08498.jpg
Saved: frame_08131_反転.jpg
Saved: frame_02652_反転.jpg
Saved: frame_01093.jpg
Saved: frame_00414.jpg
Saved: frame_05975.jpg
Saved: frame_04565_反転.jpg
Saved: frame_05975_反転.jpg
Saved: frame_07541.jpg
Saved: frame_04702.jpg
Saved: frame_05018.jpg
Saved: frame_06870_反転.jpg
Saved: frame_04138.jpg
Saved: frame_02058_反転.jpg
Saved: frame_09011_反転.jpg
Saved: frame_02402.jpg
Saved: frame_05437.jpg
Saved: frame_12771.jpg
Saved: frame_06684_反転.jpg
Saved: frame_08814_反転.jpg
Saved: frame_09516_反転.jpg
Saved: frame_08645.jpg
Saved: frame_04094.jpg
Saved: frame_04702_反転.jpg
Saved: frame_07614.jpg
Saved: frame_06308_反転.jpg
Saved: frame_04905_反転.jpg
Saved: frame_03658.jpg
Saved: frame_01093_反転.jpg
Saved: frame_07861_反転.jpg
Saved: frame_01714.jpg
Saved: frame_00185.jpg
Saved: frame_05823.jpg
Saved: frame_06870.jpg
Saved: frame_03263.jpg
Saved: frame_00795.jpg
Saved: frame_05071_反転.jpg
Saved: frame_06938.jpg
Saved: frame_07541_反転.jpg
Saved: frame_00984_反転.jpg
Saved: frame_05616_反転.jpg
Saved: frame_12347.jpg
Saved: frame_06642.jpg
Saved: frame_09065_反転.jpg
Saved: frame_07206.jpg
Saved: frame_00579.jpg
Saved: frame_08131.jpg
Saved: frame_01459.jpg
Saved: frame_09011.jpg
Saved: frame_06444.jpg
Saved: frame_06142_反転.jpg
Saved: frame_00627.jpg
Saved: frame_06033_反転.jpg
Saved: frame_00185_反転.jpg
Saved: frame_12035.jpg
Saved: frame_05126_反転.jpg
Saved: frame_01949_反転.jpg
Saved: frame_05616.jpg
Saved: frame_08442_反転.jpg
Saved: frame_04905.jpg
Saved: frame_06684.jpg
Saved: frame_04094_反転.jpg
Saved: frame_08937_反転.jpg
Saved: frame_00627_反転.jpg
Saved: frame_08442.jpg
Saved: frame_00832_反転.jpg
Saved: frame_07111_反転.jpg
Saved: frame_05774.jpg
Saved: frame_01553_反転.jpg
Saved: frame_07498_反転.jpg
Saved: frame_01418_反転.jpg
Saved: frame_06308.jpg
Saved: frame_03933.jpg
Saved: frame_06642_反転.jpg
Saved: frame_07111.jpg
Saved: frame_06938_反転.jpg
Saved: frame_00579_反転.jpg
Saved: frame_05018_反転.jpg
Saved: frame_02875.jpg
Saved: frame_00324_反転.jpg
Saved: frame_02652.jpg
Saved: frame_08814.jpg
Saved: frame_00537.jpg
Saved: frame_08355.jpg
Saved: frame_00900_反転.jpg
Saved: frame_04827_反転.jpg
Saved: frame_05437_反転.jpg
Saved: frame_00324.jpg
Saved: frame_06033.jpg
Saved: frame_03763_反転.jpg
Saved: frame_08645_反転.jpg
Saved: frame_09516.jpg
Saved: frame_12347_反転.jpg
Saved: frame_05456_反転.jpg
Saved: frame_03933_反転.jpg
Saved: frame_03626_反転.jpg
Saved: frame_07498.jpg
Saved: frame_04138_反転.jpg
Saved: frame_01459_反転.jpg
Saved: frame_08167_反転.jpg
Saved: frame_06142.jpg
Saved: frame_07206_反転.jpg
Saved: frame_09065.jpg
Saved: frame_04229_反転.jpg
Saved: frame_00900.jpg
Saved: frame_05071.jpg
Saved: frame_00795_反転.jpg
Saved: frame_04417_反転.jpg
Saved: frame_12035_反転.jpg
Saved: frame_00414_反転.jpg
Saved: frame_12295.jpg
Saved: frame_05823_反転.jpg
Saved: frame_09283_反転.jpg
Saved: frame_03763.jpg
"""