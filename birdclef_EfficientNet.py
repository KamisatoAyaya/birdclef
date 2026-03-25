import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
import cv2

# === 設定裝置 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === （資料增強）===
def add_white_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def time_shift(y, shift_max=0.2, sr=16000):
    shift = np.random.randint(int(sr * -shift_max), int(sr * shift_max))
    return np.roll(y, shift)

def pitch_shift(y, sr, pitch_range=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-pitch_range, pitch_range))

def time_stretch_waveform(y):
    rate = np.random.uniform(0.8, 1.2)
    try:
        y_stretch = librosa.effects.time_stretch(y, rate)
        return y_stretch
    except Exception as e:
        print(f"[Time Stretch Error] {e}")
        return y

def spec_augment(mel_spectrogram, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.15):
    spec = mel_spectrogram.copy()
    num_mel_channels = spec.shape[0]
    num_time_steps = spec.shape[1]

    for _ in range(num_mask):
        f = int(freq_masking_max_percentage * num_mel_channels)
        f0 = random.randint(0, num_mel_channels - f)
        spec[f0:f0 + f, :] = 0

        t = int(time_masking_max_percentage * num_time_steps)
        t0 = random.randint(0, num_time_steps - t)
        spec[:, t0:t0 + t] = 0

    return spec

def audio2melspec(audio_data, sr, n_fft, hop_length, n_mels, fmin, fmax):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        pad_mode="reflect",
        norm='slaney',
        htk=True,
        center=True,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm

class BirdDataset(Dataset):
    def __init__(self, dataframe, data_path, sr=16000, duration=5, n_mels=512, max_len=313, num_classes=206, augment=True):
        self.df = dataframe
        self.data_path = data_path
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.max_len = max_len
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_path, row['filepath'])
        y, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)

        if self.augment:
            if np.random.rand() < 0.5:
                y = add_white_noise(y)
            if np.random.rand() < 0.3:
                y = time_shift(y)
            if np.random.rand() < 0.3:
                y = pitch_shift(y, sr=self.sr)
            if np.random.rand() < 0.3:
                y = time_stretch_waveform(y)

        mel_db = audio2melspec(y, self.sr, 2048, 512, self.n_mels, 20, 16000)

        if self.augment:
            mel_db = spec_augment(mel_db)

        if mel_db.shape[1] < self.max_len:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.max_len - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :self.max_len]

        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[row['label']] = 1.0
        return mel_tensor, label

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

# === 模型定義 ===
class EfficientNetV2BirdCLEF(nn.Module):
    def __init__(self, num_classes, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_l",
            in_chans=1,
            pretrained=pretrained,
            features_only=True,
            drop_rate=dropout,
            drop_path_rate=dropout,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # 將 feature map 壓成 [B, C, 1, 1]

        total_channels = sum([self.backbone.feature_info[i]['num_chs'] for i in [-3, -2, -1]])  # 取最後三層的 channel 數量加總

        self.head = nn.Sequential(
            nn.Conv2d(total_channels, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        features = self.backbone(x)
        pooled = [self.pool(feat) for feat in features[-3:]]  # 每個都壓成 [B, C, 1, 1]
        x = torch.cat(pooled, dim=1)  # 現在可以安全 concat
        return self.head(x)


# === 訓練主流程 ===
def main():
    start_epoch = 0
    num_epochs = 10
    # === 檔案路徑設定 ===
    data_root = "D:/birdclef-2025-Efficient"
    train_audio_dir = os.path.join(data_root, "train_audio")
    metadata_path = os.path.join(data_root, "train.csv")
    checkpoint_dir = "D:/birdclef-2025-Efficient/working"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === 讀取資料與標籤處理 ===
    df = pd.read_csv(metadata_path)
    df["filepath"] = df["filename"]
    df = df[["filepath", "primary_label"]]
    df.rename(columns={"primary_label": "label"}, inplace=True)
    label_map = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
    df["label"] = df["label"].map(label_map)
    num_classes = len(label_map)

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

    train_dataset = BirdDataset(train_df, train_audio_dir, num_classes=num_classes, augment=True)
    val_dataset = BirdDataset(val_df, train_audio_dir, num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=2)

    # === 初始化模型與訓練組件 ===
    model = EfficientNetV2BirdCLEF(num_classes=num_classes, pretrained=True).to(device)
    # 你可以調整 alpha、gamma
    criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    grad_plot_dir = os.path.join(checkpoint_dir, "grad_flow")
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # === 訓練迴圈 ===
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Train] Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(train_loss=total_loss / (batch_idx + 1))

        avg_train_loss = total_loss / len(train_loader)

        # === 驗證 ===
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_targets.append(labels.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())  # sigmoid 將 logits 轉為機率

        avg_val_loss = val_loss / len(val_loader)

        # === 計算 mAP ===
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)

        val_mAP = average_precision_score(all_targets, all_outputs, average="macro")
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mAP: {val_mAP:.4f}")

        # === 儲存 Checkpoint ===
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, checkpoint_path)
        scheduler.step()
        print(f"Checkpoint saved at {checkpoint_path}")

    # === 儲存最終模型 ===
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")


if __name__ == "__main__":
    main()
