import os
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm
import cv2

# === Dataset 類別（從 .npy 檔案讀取 + 預快取進記憶體） ===
class CachedBirdNpyDataset(Dataset):
    def __init__(self, dataframe, data_path, num_classes=206):
        self.mels = []
        self.labels = []
        for i in tqdm(range(len(dataframe)), desc='Loading all mel files into memory'):
            row = dataframe.iloc[i]
            npy_name = os.path.basename(row['filepath']).replace(".ogg", ".npy")
            npy_path = os.path.join(data_path, npy_name)
            mel = np.load(npy_path)
            if mel.shape != (256, 256):
                mel = cv2.resize(mel, (256, 256), interpolation=cv2.INTER_LINEAR)
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
            label = torch.zeros(num_classes, dtype=torch.float32)
            label[row['label']] = 1.0
            self.mels.append(mel_tensor)
            self.labels.append(label)

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return self.mels[idx], self.labels[idx]

# === 模型定義 ===
class BirdCLEFModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1, num_classes=0)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x.mean(dim=[2, 3])  # Global Average Pooling
        x = self.classifier(x)
        return x

def main():
    # === 配置 ===
    data_root = "D:/birdclef-2025"
    npy_path = os.path.join(data_root, "melspec_npy")
    metadata_path = os.path.join(data_root, "train.csv")

    batch_size = 32
    num_epochs = 10
    save_path = os.path.join(data_root, "working", "convnext_final.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 資料與模型準備 ===
    metadata = pd.read_csv(metadata_path)
    metadata["filepath"] = metadata["filename"]
    df = metadata[["filepath", "primary_label"]].copy()
    df.rename(columns={"primary_label": "label"}, inplace=True)

    label_to_idx = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    df['label'] = df['label'].map(label_to_idx)
    num_classes = len(label_to_idx)

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

    train_dataset = CachedBirdNpyDataset(train_df, npy_path, num_classes=num_classes)
    val_dataset = CachedBirdNpyDataset(val_df, npy_path, num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = BirdCLEFModel("convnextv2_nano.fcmae_ft_in22k_in1k", num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler()  # ✅ AMP 混合精度初始化

    # === EarlyStopping 參數 ===
    early_stop_patience = 3
    best_val_loss = float('inf')
    early_stop_counter = 0

    # === 訓練迴圈 ===
    print("Start training ConvNeXtV2 model...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # ✅ 混合精度 forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (batch_idx + 1))

        # === 驗證 ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():  # ✅ 混合精度驗證
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")

        # === EarlyStopping ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at epoch {epoch+1}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best model saved to {save_path}")

if __name__ == "__main__":
    main()