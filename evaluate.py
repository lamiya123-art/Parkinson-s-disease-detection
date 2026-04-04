# ============================================================
# EVALUATE.PY — Final Test Evaluation
# ============================================================

import os, json, gc, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, f1_score,
                              accuracy_score, confusion_matrix,
                              roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ASTModel
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────
DRIVE_DIR  = r'C:\PCL\Pd_detection'
CHUNK_DIR  = os.path.join(DRIVE_DIR, 'chunks')
MODEL_DIR  = os.path.join(DRIVE_DIR, 'models')
PLOT_DIR   = os.path.join(DRIVE_DIR, 'plots')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

# ── Load test data ─────────────────────────────────────────
test_labels  = np.load(os.path.join(DRIVE_DIR, 'test_labels.npy'))
test_parents = np.load(os.path.join(DRIVE_DIR, 'test_parents.npy'))

all_chunks = sorted([
    os.path.join(CHUNK_DIR, f)
    for f in os.listdir(CHUNK_DIR)
    if f.endswith('.npz')
])
test_chunk_paths = [p for p in all_chunks if 'test_chunk' in p]

print(f"Test chunks : {len(test_chunk_paths)}")
print(f"Test segments: {len(test_labels)}")
print(f"Test HC: {np.sum(test_labels==0)} "
      f"PD: {np.sum(test_labels==1)}")

print("\nLoading test chunks into RAM...")
all_test_mels, all_test_mfccs = [], []
for path in tqdm(test_chunk_paths):
    chunk = np.load(path)
    all_test_mels.append(chunk['mels'])
    all_test_mfccs.append(chunk['mfccs'])
    del chunk; gc.collect()
all_test_mels  = np.concatenate(all_test_mels,  axis=0)
all_test_mfccs = np.concatenate(all_test_mfccs, axis=0)
print(f"Test mels  : {all_test_mels.shape}")
print(f"Test mfccs : {all_test_mfccs.shape}")

# ── Dataset ────────────────────────────────────────────────
class ParkinsonsDataset(Dataset):
    def __init__(self, mels, mfccs, labels):
        self.mels   = mels
        self.mfccs  = mfccs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel  = torch.tensor(self.mels[idx],
                             dtype=torch.float32).unsqueeze(0)
        mfcc = torch.tensor(self.mfccs[idx],
                             dtype=torch.float32)
        lbl  = torch.tensor(float(self.labels[idx]),
                             dtype=torch.float32)
        return mel, mfcc, lbl

# ── Model ──────────────────────────────────────────────────
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


# ── Evaluate one model ─────────────────────────────────────
def evaluate_model(model_path, test_loader):
    model = HybridPDDetector().to(device)
    model.load_state_dict(torch.load(model_path,
                                      map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, mfcc, lbl in tqdm(test_loader,
                                    desc=f"Evaluating {os.path.basename(model_path)}"):
            mel  = mel.to(device)
            mfcc = mfcc.to(device)
            ast_input = prepare_ast_input(mel)
            with torch.amp.autocast('cuda'):
                out = model(mel, mfcc, ast_input).squeeze(1)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.numpy())

    del model; gc.collect(); torch.cuda.empty_cache()
    return np.array(all_preds), np.array(all_labels)


# ── Run evaluation ─────────────────────────────────────────
test_ds     = ParkinsonsDataset(all_test_mels,
                                 all_test_mfccs,
                                 test_labels)
test_loader = DataLoader(test_ds, batch_size=8,
                          shuffle=False, num_workers=0)

fold_preds = []
model_files = sorted([
    os.path.join(MODEL_DIR, f)
    for f in os.listdir(MODEL_DIR)
    if f.endswith('.pth')
])

print(f"\nFound {len(model_files)} models: "
      f"{[os.path.basename(f) for f in model_files]}")

for model_path in model_files:
    preds, labels = evaluate_model(model_path, test_loader)
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels,
                          [1 if p>0.5 else 0 for p in preds])
    f1  = f1_score(labels,
                   [1 if p>0.5 else 0 for p in preds])
    print(f"\n{os.path.basename(model_path)}:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Acc: {acc:.4f}")
    print(f"  F1 : {f1:.4f}")
    fold_preds.append(preds)

# ── Ensemble (average predictions) ────────────────────────
print("\n" + "="*55)
print("  ENSEMBLE RESULTS (average of all folds)")
print("="*55)
ensemble_preds  = np.mean(fold_preds, axis=0)
ensemble_binary = [1 if p>0.5 else 0 for p in ensemble_preds]

final_auc = roc_auc_score(labels, ensemble_preds)
final_acc = accuracy_score(labels, ensemble_binary)
final_f1  = f1_score(labels, ensemble_binary)
final_cm  = confusion_matrix(labels, ensemble_binary)

print(f"  AUC      : {final_auc:.4f}")
print(f"  Accuracy : {final_acc:.4f}")
print(f"  F1 Score : {final_f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  TN:{final_cm[0][0]} FP:{final_cm[0][1]}")
print(f"  FN:{final_cm[1][0]} TP:{final_cm[1][1]}")

# ── Plots ──────────────────────────────────────────────────
# 1. Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['HC','PD'],
            yticklabels=['HC','PD'])
plt.title('Confusion Matrix — Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'),
            dpi=150)
plt.show()
print("✅ Saved confusion_matrix.png")

# 2. ROC Curve
fpr, tpr, _ = roc_curve(labels, ensemble_preds)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue',
         label=f'ROC Curve (AUC = {final_auc:.4f})')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Test Set')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'roc_curve.png'), dpi=150)
plt.show()
print("✅ Saved roc_curve.png")

print("\n" + "="*55)
print("  FINAL TEST RESULTS SUMMARY")
print("="*55)
print(f"  AUC      : {final_auc:.4f}")
print(f"  Accuracy : {final_acc:.4f} "
      f"({final_acc*100:.2f}%)")
print(f"  F1 Score : {final_f1:.4f}")
print(f"  Sensitivity (Recall): "
      f"{final_cm[1][1]/(final_cm[1][1]+final_cm[1][0]):.4f}")
print(f"  Specificity: "
      f"{final_cm[0][0]/(final_cm[0][0]+final_cm[0][1]):.4f}")
print("="*55)
print("\n✅ Evaluation complete!")