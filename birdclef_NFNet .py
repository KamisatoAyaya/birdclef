import os
import torch
import pandas as pd
import numpy as np
import librosa
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm

# === 資料增強函式 ===
def add_white_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def time_shift(y, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-n_steps, n_steps + 1))

def spec_augment(mel_spectrogram, max_mask_pct=0.1, num_mask=2):
    mel = mel_spectrogram.copy()
    num_mel_channels, num_time_steps = mel.shape
    for _ in range(num_mask):
        freq_mask = int(np.random.uniform(0.0, max_mask_pct) * num_mel_channels)
        freq_start = np.random.randint(0, num_mel_channels - freq_mask)
        mel[freq_start:freq_start + freq_mask, :] = 0

        time_mask = int(np.random.uniform(0.0, max_mask_pct) * num_time_steps)
        time_start = np.random.randint(0, num_time_steps - time_mask)
        mel[:, time_start:time_start + time_mask] = 0
    return mel

# === Dataset 類別 ===
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
            if np.random.rand() < 0.5: y = add_white_noise(y)
            if np.random.rand() < 0.3: y = time_shift(y)
            if np.random.rand() < 0.3: y = pitch_shift(y, sr=self.sr)

        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=2048, hop_length=512,
                                             n_mels=self.n_mels, fmin=20, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if self.augment and np.random.rand() < 0.3:
            mel_db = spec_augment(mel_db)

        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        if mel_db.shape[1] < self.max_len:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.max_len - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :self.max_len]

        mel_tensor = torch.tensor(np.stack([mel_db] * 3, axis=0), dtype=torch.float32)
        label_index = row['label']
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[label_index] = 1.0
        return mel_tensor, label

# === 主流程 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_root = "D:/birdclef-2025"
    os.makedirs(os.path.join(data_root, "working"), exist_ok=True)
    train_data_path = os.path.join(data_root, "train_audio")
    train_metadata_path = os.path.join(data_root, "train.csv")

    train_metadata = pd.read_csv(train_metadata_path)
    train_metadata["filepath"] = train_metadata["filename"]
    train_df = train_metadata[["filepath", "primary_label"]].copy()
    train_df.rename(columns={"primary_label": "label"}, inplace=True)

    label_to_idx = {label: idx for idx, label in enumerate(sorted(train_df['label'].unique()))}
    train_df['label'] = train_df['label'].map(label_to_idx)
    num_classes = len(label_to_idx)

    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

    train_dataset = BirdDataset(train_df, train_data_path, num_classes=num_classes, augment=True)
    val_dataset = BirdDataset(val_df, train_data_path, num_classes=num_classes, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    model = timm.create_model("dm_nfnet_f0", pretrained=False, num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    num_epochs = 10
    save_path = os.path.join(data_root, "working", "nfnet_checkpoint.pth")

    print("Training NFNet from scratch!")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[NFNet] Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(train_loss=running_loss / (batch_idx + 1))

        # 驗證
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {running_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, save_path)

    final_model_path = os.path.join(data_root, "working", "nfnet_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"NFNet final model saved at {final_model_path}")

if __name__ == "__main__":
    main()
