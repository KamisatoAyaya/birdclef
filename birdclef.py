import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2

# === 設定裝置 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)

def spec_augment(mel, freq_mask_param=8, time_mask_param=20):
    mel = mel.copy()
    num_mel, num_time = mel.shape
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, num_mel - f)
    mel[f0:f0+f, :] = 0
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, num_time - t)
    mel[:, t0:t0+t] = 0
    return mel

#隨機雜訊
def add_white_noise(y, noise_level=0.05):
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
    def __init__(self, dataframe, data_path, sr=16000, duration=10, n_mels=64, max_len=313, num_classes=206, transform=None, augment=True):
        self.dataframe = dataframe
        self.data_path = data_path
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.max_len = max_len
        self.num_classes = num_classes
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
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

        mel_db = audio2melspec(y, self.sr, 2048, 512, self.n_mels, 20, 8000)

        if self.augment and np.random.rand() < 0.3:
            mel_db = spec_augment(mel_db)

        if mel_db.shape[1] < self.max_len:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.max_len - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :self.max_len]

        mel_db_3ch = np.stack([mel_db] * 3, axis=0)
        mel_tensor = torch.tensor(mel_db_3ch, dtype=torch.float32)
        label_index = row['label']
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[label_index] = 1.0
        return mel_tensor, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            #layer1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            #layer2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            #layer3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            #layer4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool2d((2, 2))
            
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 2 * 2, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
  

    
def main():
    # 讀取 BirdCLEF 2025 metadata
    data_root = "D:/birdclef-2025"
    train_data_path = os.path.join(data_root, "train_audio")
    train_metadata_path = os.path.join(data_root, "train.csv")
    sample_submission_path = os.path.join(data_root, "sample_submission.csv")
    test_audio_dir = os.path.join(data_root, "test_soundscapes")

    train_metadata = pd.read_csv(train_metadata_path)
    train_metadata["filepath"] = train_metadata["filename"]
    train_df = train_metadata[["filepath", "primary_label"]].copy()
    train_df.rename(columns={"primary_label": "label"}, inplace=True)

    # Label 名稱轉數字
    label_to_idx = {label: idx for idx, label in enumerate(sorted(train_df['label'].unique()))}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    train_df['label'] = train_df['label'].map(label_to_idx)
    num_classes = len(label_to_idx)

    # 分割訓練與驗證集
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

    # 建立 Dataset 和 DataLoader
    train_dataset = BirdDataset(train_df, train_data_path, num_classes=num_classes,augment=False)
    val_dataset = BirdDataset(val_df, train_data_path, num_classes=num_classes,augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 初始化模型
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)
    #減少learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    start_epoch = 0
    num_epochs = 10
    save_path = "D:/birdclef-2025/working/latest_checkpoint.pth"    

    # 開始訓練
    print("Start training!")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Train] Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(train_loss=running_loss / (batch_idx + 1))
        avg_train_loss = running_loss / len(train_loader)

        # 驗證
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 儲存 checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, save_path)
        print(f"Saved checkpoint -> {save_path}")

    # 儲存最終模型
    final_model_path = "D:/birdclef-2025/working/final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

if __name__ == "__main__":
    main()
