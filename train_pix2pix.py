# train_pix2pix.py
# =========================
# Pix2Pix (paired image-to-image) trainer for "combined" images.
# Left half: input, Right half: target
# No torchvision. Works on Python 3.9, PyTorch + PIL + NumPy only.
# =========================

import os
import time
from typing import Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ========= ユーザー設定 =========
COMBINED_DIR = "pix2pix/combined"     # 左:入力 右:ターゲット の横連結画像が並ぶフォルダ
OUTPUT_DIR   = "pix2pix/outputs"      # 学習過程のサンプル画像など
CKPT_PATH    = "pix2pix_generator.pth"  # 学習済みGeneratorの保存先
IMG_SIZE     = (256, 256)             # 学習解像度（固定）
EPOCHS       = 100
BATCH_SIZE   = 1                      # pix2pix論文は1がデフォ
LR           = 2e-4
L1_LAMBDA    = 100.0                  # 再構成L1の重み
SAVE_EVERY   = 1                      # 何epochごとにサンプル保存するか
SEED         = 42                     # 再現性用（軽め）

# ========= 最低限の前処理/保存（torchvision不使用） =========
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> torch.FloatTensor [3,H,W], range [0,1]"""
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # グレースケールは3chへ
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def resize_pil(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.BICUBIC)

def save_triplet(inp: torch.Tensor, fake: torch.Tensor, tar: torch.Tensor, path: str):
    """
    inp, fake, tar: [-1,1] の [1,3,H,W] もしくは [3,H,W]
    横に結合してPNG保存
    """
    if inp.dim() == 4:
        inp, fake, tar = inp[0], fake[0], tar[0]
    def denorm(x):  # [-1,1] -> [0,1]
        return (x * 0.5 + 0.5).clamp(0, 1)
    inp_img  = denorm(inp).permute(1, 2, 0).cpu().numpy()
    fake_img = denorm(fake).permute(1, 2, 0).cpu().numpy()
    tar_img  = denorm(tar).permute(1, 2, 0).cpu().numpy()
    concat = np.concatenate([inp_img, fake_img, tar_img], axis=1)
    Image.fromarray((concat * 255).astype(np.uint8)).save(path)

# ========= Dataset（combined形式） =========
class CombinedDataset(Dataset):
    def __init__(self, root: str, size: Tuple[int, int] = (256, 256)):
        self.root = root
        self.files = sorted([f for f in os.listdir(root)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if not self.files:
            raise RuntimeError(f"No images found in: {root}")
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root, self.files[idx])
        img = Image.open(path).convert("RGB")
        w, h = img.size
        w2 = w // 2
        left  = img.crop((0,   0,  w2, h))  # 入力
        right = img.crop((w2,  0,  w,  h))  # ターゲット

        # リサイズ -> テンソル化 -> [-1,1] 正規化
        left  = pil_to_tensor(resize_pil(left,  self.size)) * 2 - 1
        right = pil_to_tensor(resize_pil(right, self.size)) * 2 - 1
        return left, right

# ========= U-Net Generator（8段, InstanceNorm, down1/down8は正規化なし） =========
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_dropout=False, norm='in'):
        """
        norm: 'in' or 'none'
        """
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

    def forward(self, x):
        return self.block(x)

class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.down1 = UNetBlock(3,    64,  down=True,  norm='none')  # 256->128
        self.down2 = UNetBlock(64,   128, down=True,  norm='in')    # 128->64
        self.down3 = UNetBlock(128,  256, down=True,  norm='in')    # 64 ->32
        self.down4 = UNetBlock(256,  512, down=True,  norm='in')    # 32 ->16
        self.down5 = UNetBlock(512,  512, down=True,  norm='in')    # 16 ->8
        self.down6 = UNetBlock(512,  512, down=True,  norm='in')    # 8  ->4
        self.down7 = UNetBlock(512,  512, down=True,  norm='in')    # 4  ->2
        self.down8 = UNetBlock(512,  512, down=True,  norm='none')  # 2  ->1（正規化なし）

        # 1 -> 2（以降スキップとconcat）
        self.bottom = UNetBlock(512, 512, down=False, norm='in')

        self.up7 = UNetBlock(1024, 512, down=False, use_dropout=True, norm='in')  # 2->4
        self.up6 = UNetBlock(1024, 512, down=False, use_dropout=True, norm='in')  # 4->8
        self.up5 = UNetBlock(1024, 512, down=False, use_dropout=True, norm='in')  # 8->16
        self.up4 = UNetBlock(1024, 256, down=False, norm='in')                    #16->32
        self.up3 = UNetBlock(512,  128, down=False, norm='in')                    #32->64
        self.up2 = UNetBlock(256,  64,  down=False, norm='in')                    #64->128
        self.final = nn.Sequential(                                               #128->256
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)   # 128
        d2 = self.down2(d1)  # 64
        d3 = self.down3(d2)  # 32
        d4 = self.down4(d3)  # 16
        d5 = self.down5(d4)  # 8
        d6 = self.down6(d5)  # 4
        d7 = self.down7(d6)  # 2
        d8 = self.down8(d7)  # 1

        b  = self.bottom(d8)                          # 2
        u7 = self.up7(torch.cat([b,  d7], dim=1))     # 4
        u6 = self.up6(torch.cat([u7, d6], dim=1))     # 8
        u5 = self.up5(torch.cat([u6, d5], dim=1))     # 16
        u4 = self.up4(torch.cat([u5, d4], dim=1))     # 32
        u3 = self.up3(torch.cat([u4, d3], dim=1))     # 64
        u2 = self.up2(torch.cat([u3, d2], dim=1))     # 128
        out = self.final(torch.cat([u2, d1], dim=1))  # 256
        return out

# ========= PatchGAN Discriminator（最初の層は正規化なし） =========
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_c, out_c, use_norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if use_norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True, track_running_stats=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels * 2, 64,  use_norm=False),  # 入力とターゲットをconcat
            *block(64, 128,  use_norm=True),
            *block(128, 256, use_norm=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# ========= 学習ループ =========
def main():
    # 再現性（軽め）
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # デバイス
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # データ
    dataset = CombinedDataset(COMBINED_DIR, size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # モデル・損失・最適化
    G = GeneratorUNet().to(device)
    D = PatchDiscriminator().to(device)
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    # 学習
    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        t0 = time.time()
        for inp, tar in loader:
            inp, tar = inp.to(device), tar.to(device)

            # ---- D step ----
            opt_D.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_detached = G(inp)
            pred_real = D(inp, tar)
            pred_fake = D(inp, fake_detached)
            loss_D = 0.5 * (
                criterion_gan(pred_real, torch.ones_like(pred_real)) +
                criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            )
            loss_D.backward()
            opt_D.step()

            # ---- G step ----
            opt_G.zero_grad(set_to_none=True)
            fake = G(inp)
            pred_fake_for_G = D(inp, fake)
            loss_G_gan = criterion_gan(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G_l1  = criterion_l1(fake, tar) * L1_LAMBDA
            loss_G = loss_G_gan + loss_G_l1
            loss_G.backward()
            opt_G.step()

        dt = time.time() - t0
        print(f"Epoch [{epoch}/{EPOCHS}]  "
              f"Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}  "
              f"(L1: {loss_G_l1.item():.4f}, GAN: {loss_G_gan.item():.4f})  "
              f"{dt:.1f}s")

        # 進捗サンプル保存
        if epoch % SAVE_EVERY == 0:
            save_triplet(inp.detach().cpu(), fake.detach().cpu(), tar.detach().cpu(),
                         os.path.join(OUTPUT_DIR, f"epoch_{epoch:03d}.png"))

    # Generator保存
    torch.save(G.state_dict(), CKPT_PATH)
    print(f"✅ Training finished. Saved: {CKPT_PATH}")

if __name__ == "__main__":
    main()

#出力結果 10/15
'''
 python3 train_pix2pix.py
Device: mps
Epoch [1/100]  Loss_D: 0.7569  Loss_G: 10.0280  (L1: 7.7447, GAN: 2.2834)  32.0s
Epoch [2/100]  Loss_D: 0.5185  Loss_G: 4.1915  (L1: 3.3115, GAN: 0.8800)  27.0s
Epoch [3/100]  Loss_D: 0.6762  Loss_G: 1.7638  (L1: 0.9128, GAN: 0.8511)  27.0s
Epoch [4/100]  Loss_D: 0.5338  Loss_G: 2.9824  (L1: 0.8694, GAN: 2.1129)  27.0s
Epoch [5/100]  Loss_D: 0.4679  Loss_G: 3.3948  (L1: 2.5976, GAN: 0.7972)  27.3s
Epoch [6/100]  Loss_D: 0.2007  Loss_G: 5.8220  (L1: 3.8130, GAN: 2.0090)  27.2s
Epoch [7/100]  Loss_D: 0.3776  Loss_G: 3.8219  (L1: 2.4510, GAN: 1.3709)  27.0s
Epoch [8/100]  Loss_D: 0.5613  Loss_G: 3.0725  (L1: 0.9037, GAN: 2.1688)  26.9s
Epoch [9/100]  Loss_D: 0.5808  Loss_G: 2.4760  (L1: 1.7862, GAN: 0.6898)  26.9s
Epoch [10/100]  Loss_D: 0.5600  Loss_G: 1.6135  (L1: 0.8240, GAN: 0.7895)  26.9s
Epoch [11/100]  Loss_D: 0.4385  Loss_G: 2.9154  (L1: 0.6609, GAN: 2.2545)  26.9s
Epoch [12/100]  Loss_D: 0.1600  Loss_G: 4.9602  (L1: 0.4524, GAN: 4.5077)  26.9s
Epoch [13/100]  Loss_D: 0.3245  Loss_G: 4.0025  (L1: 1.9401, GAN: 2.0624)  26.9s
Epoch [14/100]  Loss_D: 0.3515  Loss_G: 3.3469  (L1: 0.5992, GAN: 2.7478)  26.9s
Epoch [15/100]  Loss_D: 0.0991  Loss_G: 11.4451  (L1: 7.7549, GAN: 3.6902)  26.9s
Epoch [16/100]  Loss_D: 0.3826  Loss_G: 4.9768  (L1: 2.8497, GAN: 2.1270)  27.1s
Epoch [17/100]  Loss_D: 0.0434  Loss_G: 8.0224  (L1: 2.3997, GAN: 5.6228)  26.9s
Epoch [18/100]  Loss_D: 0.0119  Loss_G: 6.2286  (L1: 0.7066, GAN: 5.5220)  26.9s
Epoch [19/100]  Loss_D: 0.0068  Loss_G: 6.5186  (L1: 0.9090, GAN: 5.6096)  26.9s
Epoch [20/100]  Loss_D: 0.0090  Loss_G: 6.5200  (L1: 0.6022, GAN: 5.9178)  26.9s
Epoch [21/100]  Loss_D: 0.0048  Loss_G: 6.4850  (L1: 0.6491, GAN: 5.8359)  26.9s
Epoch [22/100]  Loss_D: 0.2581  Loss_G: 10.1586  (L1: 5.7677, GAN: 4.3909)  26.9s
Epoch [23/100]  Loss_D: 0.0057  Loss_G: 7.6854  (L1: 2.1489, GAN: 5.5365)  26.9s
Epoch [24/100]  Loss_D: 0.0034  Loss_G: 7.2144  (L1: 0.7014, GAN: 6.5130)  27.2s
Epoch [25/100]  Loss_D: 0.1850  Loss_G: 2.3900  (L1: 0.3689, GAN: 2.0211)  27.2s
Epoch [26/100]  Loss_D: 0.0390  Loss_G: 7.0383  (L1: 2.0676, GAN: 4.9707)  27.0s
Epoch [27/100]  Loss_D: 0.0033  Loss_G: 12.6761  (L1: 7.0149, GAN: 5.6613)  27.2s
Epoch [28/100]  Loss_D: 0.0130  Loss_G: 10.1050  (L1: 5.5804, GAN: 4.5246)  27.2s
Epoch [29/100]  Loss_D: 0.0050  Loss_G: 7.0435  (L1: 0.4292, GAN: 6.6143)  27.2s
Epoch [30/100]  Loss_D: 0.0038  Loss_G: 7.0138  (L1: 0.5449, GAN: 6.4689)  27.1s
Epoch [31/100]  Loss_D: 0.0009  Loss_G: 9.2068  (L1: 1.7209, GAN: 7.4860)  27.1s
Epoch [32/100]  Loss_D: 0.0031  Loss_G: 7.0982  (L1: 0.5659, GAN: 6.5323)  27.1s
Epoch [33/100]  Loss_D: 0.0045  Loss_G: 7.8105  (L1: 0.3813, GAN: 7.4292)  27.1s
Epoch [34/100]  Loss_D: 0.0008  Loss_G: 9.4853  (L1: 2.0208, GAN: 7.4646)  27.0s
Epoch [35/100]  Loss_D: 0.0030  Loss_G: 6.6090  (L1: 0.5965, GAN: 6.0125)  27.0s
Epoch [36/100]  Loss_D: 0.0071  Loss_G: 5.7702  (L1: 0.7002, GAN: 5.0700)  27.0s
Epoch [37/100]  Loss_D: 0.0019  Loss_G: 8.0454  (L1: 0.4385, GAN: 7.6069)  27.0s
Epoch [38/100]  Loss_D: 0.0043  Loss_G: 6.7624  (L1: 1.4271, GAN: 5.3353)  27.0s
Epoch [39/100]  Loss_D: 0.0008  Loss_G: 17.4501  (L1: 10.4313, GAN: 7.0189)  27.0s
Epoch [40/100]  Loss_D: 0.0012  Loss_G: 8.3616  (L1: 0.7823, GAN: 7.5794)  27.0s
Epoch [41/100]  Loss_D: 0.4709  Loss_G: 7.0321  (L1: 1.7841, GAN: 5.2480)  27.0s
Epoch [42/100]  Loss_D: 0.0031  Loss_G: 7.1055  (L1: 0.4483, GAN: 6.6572)  27.0s
Epoch [43/100]  Loss_D: 0.0246  Loss_G: 8.5069  (L1: 6.9747, GAN: 1.5322)  27.4s
Epoch [44/100]  Loss_D: 0.0007  Loss_G: 14.8728  (L1: 7.2575, GAN: 7.6154)  27.0s
Epoch [45/100]  Loss_D: 0.0004  Loss_G: 10.2292  (L1: 1.7654, GAN: 8.4638)  27.0s
Epoch [46/100]  Loss_D: 0.0003  Loss_G: 13.8492  (L1: 4.9908, GAN: 8.8584)  27.1s
Epoch [47/100]  Loss_D: 0.0004  Loss_G: 16.4051  (L1: 8.5535, GAN: 7.8517)  27.0s
Epoch [48/100]  Loss_D: 0.0013  Loss_G: 7.2750  (L1: 0.6778, GAN: 6.5972)  27.1s
Epoch [49/100]  Loss_D: 0.0019  Loss_G: 9.3514  (L1: 0.3799, GAN: 8.9714)  27.1s
Epoch [50/100]  Loss_D: 0.1606  Loss_G: 4.3905  (L1: 0.4473, GAN: 3.9431)  27.1s
Epoch [51/100]  Loss_D: 0.0002  Loss_G: 11.0844  (L1: 1.5192, GAN: 9.5651)  27.2s
Epoch [52/100]  Loss_D: 0.0010  Loss_G: 9.6889  (L1: 0.4371, GAN: 9.2518)  27.0s
Epoch [53/100]  Loss_D: 0.0032  Loss_G: 12.2585  (L1: 5.0051, GAN: 7.2534)  27.1s
Epoch [54/100]  Loss_D: 0.0003  Loss_G: 9.7979  (L1: 0.8979, GAN: 8.9000)  27.3s
Epoch [55/100]  Loss_D: 0.0002  Loss_G: 18.4362  (L1: 9.5228, GAN: 8.9135)  27.7s
Epoch [56/100]  Loss_D: 0.0004  Loss_G: 8.9774  (L1: 0.5950, GAN: 8.3825)  27.5s
Epoch [57/100]  Loss_D: 0.0003  Loss_G: 11.9105  (L1: 1.7693, GAN: 10.1411)  27.5s
Epoch [58/100]  Loss_D: 0.0004  Loss_G: 21.2481  (L1: 12.8817, GAN: 8.3663)  28.0s
Epoch [59/100]  Loss_D: 0.0298  Loss_G: 10.1045  (L1: 1.6239, GAN: 8.4806)  27.2s
Epoch [60/100]  Loss_D: 0.1780  Loss_G: 5.6487  (L1: 0.3915, GAN: 5.2572)  27.0s
Epoch [61/100]  Loss_D: 0.0007  Loss_G: 9.2257  (L1: 0.5628, GAN: 8.6629)  27.0s
Epoch [62/100]  Loss_D: 0.0006  Loss_G: 11.0453  (L1: 1.3066, GAN: 9.7388)  27.0s
Epoch [63/100]  Loss_D: 0.2140  Loss_G: 31.0880  (L1: 27.2774, GAN: 3.8106)  27.0s
Epoch [64/100]  Loss_D: 0.0027  Loss_G: 10.3550  (L1: 0.3795, GAN: 9.9755)  27.0s
Epoch [65/100]  Loss_D: 0.0011  Loss_G: 17.0568  (L1: 10.0677, GAN: 6.9890)  27.0s
Epoch [66/100]  Loss_D: 0.0002  Loss_G: 10.8045  (L1: 1.7913, GAN: 9.0132)  27.0s
Epoch [67/100]  Loss_D: 0.0025  Loss_G: 9.8149  (L1: 1.7608, GAN: 8.0541)  27.0s
Epoch [68/100]  Loss_D: 0.0005  Loss_G: 9.4939  (L1: 0.9054, GAN: 8.5885)  27.0s
Epoch [69/100]  Loss_D: 0.0005  Loss_G: 14.0509  (L1: 6.2986, GAN: 7.7523)  27.0s
Epoch [70/100]  Loss_D: 0.0003  Loss_G: 10.0836  (L1: 1.7824, GAN: 8.3012)  27.0s
Epoch [71/100]  Loss_D: 0.0002  Loss_G: 10.9092  (L1: 1.5192, GAN: 9.3900)  27.0s
Epoch [72/100]  Loss_D: 0.0003  Loss_G: 9.3695  (L1: 0.5844, GAN: 8.7851)  27.0s
Epoch [73/100]  Loss_D: 0.0002  Loss_G: 17.2926  (L1: 7.0030, GAN: 10.2897)  27.0s
Epoch [74/100]  Loss_D: 0.0004  Loss_G: 10.0390  (L1: 1.3755, GAN: 8.6634)  27.1s
Epoch [75/100]  Loss_D: 0.0002  Loss_G: 18.6605  (L1: 9.5227, GAN: 9.1378)  27.0s
Epoch [76/100]  Loss_D: 0.0077  Loss_G: 5.8696  (L1: 0.6423, GAN: 5.2273)  27.0s
Epoch [77/100]  Loss_D: 0.0009  Loss_G: 9.5845  (L1: 0.3265, GAN: 9.2580)  27.0s
Epoch [78/100]  Loss_D: 0.0002  Loss_G: 14.3110  (L1: 4.1704, GAN: 10.1406)  27.6s
Epoch [79/100]  Loss_D: 0.0001  Loss_G: 12.5589  (L1: 1.2772, GAN: 11.2817)  27.4s
Epoch [80/100]  Loss_D: 0.0002  Loss_G: 11.6577  (L1: 0.8871, GAN: 10.7706)  27.4s
Epoch [81/100]  Loss_D: 0.0005  Loss_G: 17.8701  (L1: 7.0029, GAN: 10.8672)  27.4s
Epoch [82/100]  Loss_D: 0.0002  Loss_G: 15.2048  (L1: 5.5779, GAN: 9.6269)  27.7s
Epoch [83/100]  Loss_D: 0.0005  Loss_G: 9.2756  (L1: 0.4244, GAN: 8.8512)  27.5s
Epoch [84/100]  Loss_D: 0.0001  Loss_G: 14.6782  (L1: 4.6759, GAN: 10.0023)  26.9s
Epoch [85/100]  Loss_D: 0.0001  Loss_G: 15.3131  (L1: 4.9492, GAN: 10.3640)  26.8s
Epoch [86/100]  Loss_D: 0.0001  Loss_G: 14.1763  (L1: 3.4356, GAN: 10.7407)  26.8s
Epoch [87/100]  Loss_D: 0.0001  Loss_G: 11.5207  (L1: 1.2056, GAN: 10.3151)  26.8s
Epoch [88/100]  Loss_D: 0.0002  Loss_G: 15.5128  (L1: 5.7632, GAN: 9.7496)  26.8s
Epoch [89/100]  Loss_D: 0.0003  Loss_G: 9.3529  (L1: 0.4243, GAN: 8.9285)  26.8s
Epoch [90/100]  Loss_D: 0.0004  Loss_G: 11.5535  (L1: 3.4356, GAN: 8.1179)  26.8s
Epoch [91/100]  Loss_D: 0.0001  Loss_G: 13.6236  (L1: 2.0599, GAN: 11.5636)  26.8s
Epoch [92/100]  Loss_D: 0.0001  Loss_G: 12.9983  (L1: 1.7388, GAN: 11.2595)  26.8s
Epoch [93/100]  Loss_D: 0.0004  Loss_G: 11.3138  (L1: 0.3913, GAN: 10.9225)  27.3s
Epoch [94/100]  Loss_D: 0.0002  Loss_G: 11.1398  (L1: 1.3755, GAN: 9.7643)  27.0s
Epoch [95/100]  Loss_D: 0.0001  Loss_G: 13.4621  (L1: 1.8234, GAN: 11.6386)  27.2s
Epoch [96/100]  Loss_D: 0.0001  Loss_G: 11.8919  (L1: 1.5192, GAN: 10.3728)  27.2s
Epoch [97/100]  Loss_D: 0.0014  Loss_G: 9.5080  (L1: 0.3630, GAN: 9.1451)  27.2s
Epoch [98/100]  Loss_D: 0.9747  Loss_G: 9.4595  (L1: 8.7457, GAN: 0.7138)  27.2s
Epoch [99/100]  Loss_D: 0.4629  Loss_G: 4.4701  (L1: 3.3787, GAN: 1.0915)  27.2s
Epoch [100/100]  Loss_D: 0.6171  Loss_G: 2.0559  (L1: 1.2359, GAN: 0.8200)  27.2s
✅ Training finished. Saved: pix2pix_generator.pth
'''