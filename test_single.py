# ============================================================
# TEST_SINGLE.PY — Test on a new audio file
# ============================================================

import os, gc, json
import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import ASTModel

# ── Paths ─────────────────────────────────────────────────
DRIVE_DIR  = r'C:\PCL\Pd_detection'
MODEL_DIR  = os.path.join(DRIVE_DIR, 'models')

# ── Settings ──────────────────────────────────────────────
SR          = 16000
SEG_SAMPLES = 48000
N_MFCC      = 40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load norm stats ───────────────────────────────────────
with open(os.path.join(DRIVE_DIR, 'norm_stats.json')) as f:
    norm_stats = json.load(f)

mel_mean  = np.array(norm_stats['mel_mean'])
mel_std   = np.array(norm_stats['mel_std'])
mfcc_mean = np.array(norm_stats['mfcc_mean']).reshape(-1, 1)
mfcc_std  = np.array(norm_stats['mfcc_std']).reshape(-1, 1)

# ── Model ─────────────────────────────────────────────────
class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(256,256),
            nn.ReLU(), nn.Dropout(0.3))

    def forward(self, x):
        return self.fc(self.conv(x))


class BiLSTMBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(120, 128, num_layers=2,
                            batch_first=True, dropout=0.3,
                            bidirectional=True)
        self.fc   = nn.Sequential(
            nn.Linear(256,256), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class ASTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593")
        for param in self.ast.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(768,256), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, x):
        with torch.no_grad():
            out = self.ast(input_values=x)
        return self.fc(out.last_hidden_state[:, 0, :])


class HybridPDDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_branch    = CNNBranch()
        self.bilstm_branch = BiLSTMBranch()
        self.ast_branch    = ASTBranch()
        self.attention     = nn.MultiheadAttention(
            256, num_heads=4, dropout=0.1, batch_first=True)
        self.classifier    = nn.Sequential(
            nn.Linear(768,512), nn.LayerNorm(512),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512,128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128,1))

    def forward(self, mel, mfcc, ast_in):
        c = self.cnn_branch(mel)
        b = self.bilstm_branch(mfcc)
        a = self.ast_branch(ast_in)
        fused   = torch.stack([c,b,a], dim=1)
        attn, _ = self.attention(fused, fused, fused)
        return self.classifier(attn.reshape(attn.size(0), -1))


def prepare_ast_input(mels_batch):
    mels = mels_batch.squeeze(1)
    return torch.nn.functional.interpolate(
        mels.unsqueeze(1),
        size=(1024, 128),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)


# ── Feature extraction ────────────────────────────────────
def extract_features(audio_path):
    print(f"\nLoading: {audio_path}")
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    print(f"Duration: {len(y)/SR:.1f} seconds")

    # Pad or trim to 3 seconds
    if len(y) < SEG_SAMPLES:
        y = np.pad(y, (0, SEG_SAMPLES - len(y)))
    else:
        y = y[:SEG_SAMPLES]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=128, fmax=8000)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel_mean) / (mel_std + 1e-8)
    if mel.shape[1] < 128:
        mel = np.pad(mel, ((0,0),(0,128-mel.shape[1])))
    else:
        mel = mel[:, :128]

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc   = np.concatenate([mfcc, delta, delta2], axis=0)
    mfcc   = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
    if mfcc.shape[1] < 128:
        mfcc = np.pad(mfcc, ((0,0),(0,128-mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :128]
    mfcc = mfcc.T  # (128, 120)

    return mel, mfcc


# ── Predict ───────────────────────────────────────────────
def predict(audio_path):
    mel, mfcc = extract_features(audio_path)

    mel_t  = torch.tensor(mel,  dtype=torch.float32
                           ).unsqueeze(0).unsqueeze(0).to(device)
    mfcc_t = torch.tensor(mfcc, dtype=torch.float32
                           ).unsqueeze(0).to(device)
    ast_in = prepare_ast_input(mel_t)

    model_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.endswith('.pth')
    ])

    print(f"\nRunning through {len(model_files)} models...")
    all_probs = []
    for model_path in model_files:
        model = HybridPDDetector().to(device)
        model.load_state_dict(torch.load(model_path,
                              map_location=device))
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = model(mel_t, mfcc_t, ast_in)
        prob = torch.sigmoid(out).item()
        all_probs.append(prob)
        print(f"  {os.path.basename(model_path)}: "
              f"PD probability = {prob:.4f}")
        del model; gc.collect()
        torch.cuda.empty_cache()

    ensemble_prob = np.mean(all_probs)
    prediction    = ("PARKINSON'S DETECTED"
                     if ensemble_prob > 0.5
                     else "HEALTHY CONTROL")

    print(f"\n{'='*45}")
    print(f"  RESULT: {prediction}")
    print(f"  PD Probability : {ensemble_prob:.4f} "
          f"({ensemble_prob*100:.1f}%)")
    print(f"  HC Probability : {1-ensemble_prob:.4f} "
          f"({(1-ensemble_prob)*100:.1f}%)")
    print(f"{'='*45}")
    return ensemble_prob, prediction


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    AUDIO_FILE = r'C:\PCL\test_audio.wav'

    if not os.path.exists(AUDIO_FILE):
        print(f"❌ File not found: {AUDIO_FILE}")
        print("Place a .wav audio file at C:\\PCL\\test_audio.wav")
    else:
        predict(AUDIO_FILE)