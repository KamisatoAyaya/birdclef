# === 套件載入 ===
import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm
from typing import Union

# === 設定裝置 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === 資料與標籤設定 ===
data_root = "D:/birdclef-2025-Efficient"
test_audio_dir = os.path.join(data_root, "test_soundscapes")
sample_submission_path = os.path.join(data_root, "sample_submission.csv")
sample_submission = pd.read_csv(sample_submission_path)
class_labels = sample_submission.columns[1:].tolist()
file_list = sorted([f for f in os.listdir(test_audio_dir) if f.endswith('.ogg')])

# === 音訊轉 mel spectrogram ===
def audio_to_mel(path, offset=0, duration=5, sr=32000, n_mels=128):
    y, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel_tensor.to(device)

# === 4. 定義簡單 CNN 模型 ===
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            #layer1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            #layer2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            #layer3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            #layer4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1,1))
            
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
# === 模型定義 ===
class EfficientNetV2BirdCLEF(nn.Module):
    def __init__(self, num_classes, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(
            "eca_nfnet_l0",
            in_chans=1,
            pretrained=pretrained,
            features_only=True,
            drop_rate=dropout,
            drop_path_rate=dropout,
        )
        self.head = nn.Sequential(
            nn.Conv2d(2304, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.backbone(x)[-1]  # <- 最後一層 feature map (B, 2304, H, W)
        return self.head(x)       # <- 輸出: (B, 206)


# === NFNet (TimmSED) 架構 ===
def apply_power_to_low_ranked_cols(p: np.ndarray, top_k=30, exponent=2, inplace=True) -> np.ndarray:
    if not inplace:
        p = p.copy()
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x, ratio):
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output

class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1, n_mels=24):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(n_mels)

        base_model = timm.create_model(
            base_model_name, pretrained=pretrained, in_chans=in_channels)
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.num_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block2 = AttBlockV2(
            in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        

    def forward(self, input_data):
        x = input_data.transpose(2,3)
        x = torch.cat((x,x,x),1)

        x = x.transpose(2, 3)

        x = self.encoder(x)
        
        x = torch.mean(x, dim=2)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block2(x)
        logit = torch.sum(norm_att * self.att_block2.cla(x), dim=2)

        output_dict = {
            'logit': logit,
        }

        return output_dict





# === 載入模型 ===
model_cnn = SimpleCNN(num_classes=len(class_labels)).to(device)
model_effnet = EfficientNetV2BirdCLEF(num_classes=len(class_labels)).to(device)
print(model_effnet)

model_cnn_path = r"D:/birdclef-2025-Efficient/model/birdCLEF_simpleCNN.pth"
model_effnet_path = r"D:/birdclef-2025-Efficient/model/BirdCLEF_efficientnet.pth"

model_cnn.load_state_dict(torch.load(model_cnn_path, map_location=device))
model_effnet.load_state_dict(torch.load(model_effnet_path, map_location=device))


model_cnn.eval()
model_effnet.eval()

# === 推論單一模型 ===
def predict_model(model, name="model", segment_sec=5, max_duration=60):
    model.eval()
    pred = {'row_id': []}
    for lbl in class_labels:
        pred[lbl] = []

    for fname in tqdm(file_list, desc=f"Predicting with {name}"):
        path = os.path.join(test_audio_dir, fname)

        for offset in range(0, max_duration, segment_sec):
            mel = audio_to_mel(path, offset=offset, duration=segment_sec)
            with torch.inference_mode():
                out = torch.sigmoid(model(mel)).cpu().numpy()[0]
            row_id = fname.replace('.ogg', f'_{offset + segment_sec}')
            pred['row_id'].append(row_id)
            for i, label in enumerate(class_labels):
                pred[label].append(out[i])
    
    return pd.DataFrame(pred)

# === NFNet 推論（三模型平均）===
def predict_nfnet_ensemble(segment_sec=5, max_duration=60):
    base_model_name = 'eca_nfnet_l0'
    pretrained = False
    in_channels = 3
    n_mels = 128
    model_paths = [f"D:/birdclef-2025-Efficient/sed-models/sed{i}.pth" for i in range(3)]

    sed_models = []
    for path in model_paths:
        model = TimmSED(base_model_name, num_classes=len(class_labels),
                        in_channels=in_channels, pretrained=pretrained,
                        n_mels=n_mels).to(device)
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        model.eval()
        sed_models.append(model)

    pred = {'row_id': []}
    for lbl in class_labels:
        pred[lbl] = []

    for fname in tqdm(file_list, desc="Predicting with NFNet (3 SED models)"):
        path = os.path.join(test_audio_dir, fname)

        for offset in range(0, max_duration, segment_sec):
            mel = audio_to_mel(path, offset=offset, duration=segment_sec)
            with torch.inference_mode():
                out_sum = None
                for model in sed_models:
                    p = model(mel)
                    p = torch.sigmoid(p['logit']).detach().cpu().numpy()
                    p = apply_power_to_low_ranked_cols(p, top_k=30, exponent=2)
                    out_sum = p if out_sum is None else out_sum + p
                p_avg = out_sum / len(sed_models)

            row_id = fname.replace('.ogg', f'_{offset + segment_sec}')
            pred['row_id'].append(row_id)
            for i, label in enumerate(class_labels):
                pred[label].append(p_avg[0][i])

    return pd.DataFrame(pred)

# === 執行三模型推論與融合 ===
# 加這段來讀取 sample_submission.csv
sample_submission_path = os.path.join(data_root, "sample_submission.csv")
sample_df = pd.read_csv(sample_submission_path)

df_cnn = predict_model(model_cnn, "CNN")
df_eff = predict_model(model_effnet, "EfficientNet")
df_nf = predict_nfnet_ensemble()

df_final = df_cnn.copy()
for label in class_labels:
    df_final[label] = 0.1 * df_cnn[label] + 0.2 * df_eff[label] + 0.7 * df_nf[label]

# 強制填補 NaN 為 0，並依照 sample_submission 欄位順序整理
df_final.fillna(0, inplace=True)
df_final = df_final[sample_df.columns]  # 保證格式一致
df_final.to_csv("submission.csv", index=False)
