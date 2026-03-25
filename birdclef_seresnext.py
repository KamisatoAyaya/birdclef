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

# === 主流程中先決定 device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset 類別（從 .npy 檔案讀取） ===
class BirdNpyDataset(Dataset):
    def __init__(self, dataframe, data_path, num_classes=206, max_len=313, augment=True):
        self.dataframe = dataframe
        self.data_path = data_path
        self.num_classes = num_classes
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        npy_name = os.path.basename(row['filepath']).replace(".ogg", ".npy")
        npy_path = os.path.join(self.data_path, npy_name)

        mel = np.load(npy_path)

        if mel.shape[1] < self.max_len:
            pad_width = self.max_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel = mel[:, :self.max_len]

        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        label_index = row['label']
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[label_index] = 1.0
        return mel_tensor, label


def main():
    data_root = "D:/birdclef-2025"
    npy_path = os.path.join(data_root, "melspec_npy")
    os.makedirs(os.path.join(data_root, "working"), exist_ok=True)
    metadata_path = os.path.join(data_root, "train.csv")

    metadata = pd.read_csv(metadata_path)
    metadata["filepath"] = metadata["filename"]
    df = metadata[["filepath", "primary_label"]].copy()
    df.rename(columns={"primary_label": "label"}, inplace=True)

    label_to_idx = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    df['label'] = df['label'].map(label_to_idx)
    num_classes = len(label_to_idx)

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

    train_dataset = BirdNpyDataset(train_df, npy_path, num_classes=num_classes, augment=True)
    val_dataset = BirdNpyDataset(val_df, npy_path, num_classes=num_classes, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    model = timm.create_model("seresnext50_32x4d", pretrained=False, num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    num_epochs = 10
    save_path = os.path.join(data_root, "working", "seresnext_checkpoint.pth")

    print("Training SE-ResNeXt using pre-generated mel spectrograms!")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[SEResNeXt] Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(train_loss=running_loss / (batch_idx + 1))

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

    final_model_path = os.path.join(data_root, "working", "seresnext_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"SE-ResNeXt final model saved at {final_model_path}")


if __name__ == "__main__":
    main()