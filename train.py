# ============================================================
# TRAIN.PY — Train folds 2 and 3 on RTX 2050
# ============================================================

import os, json, gc, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from transformers import ASTModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────
DRIVE_DIR  = r'C:\PCL\Pd_detection'
CHUNK_DIR  = os.path.join(DRIVE_DIR, 'chunks')
MODEL_DIR  = os.path.join(DRIVE_DIR, 'models')
PLOT_DIR   = os.path.join(DRIVE_DIR, 'plots')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ── Settings ─────────────────────────────────────────────────
BATCH_SIZE  = 8   # small for 4GB VRAM
MAX_EPOCHS  = 30
LR          = 2e-4
WEIGHT_DECAY= 1e-4
PATIENCE    = 7
N_FOLDS     = 3
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── Load labels ──────────────────────────────────────────────
tv_labels    = np.load(os.path.join(DRIVE_DIR, 'tv_labels.npy'))
tv_parents   = np.load(os.path.join(DRIVE_DIR, 'tv_parents.npy'))
test_labels  = np.load(os.path.join(DRIVE_DIR, 'test_labels.npy'))
test_parents = np.load(os.path.join(DRIVE_DIR, 'test_parents.npy'))

# ── Fix chunk paths for Windows ──────────────────────────────
all_chunks = sorted([
    os.path.join(CHUNK_DIR, f)
    for f in os.listdir(CHUNK_DIR)
    if f.endswith('.npz')
])
tv_chunk_paths   = [p for p in all_chunks if 'tv_chunk'   in p]
test_chunk_paths = [p for p in all_chunks if 'test_chunk' in p]

print(f"TV chunks   : {len(tv_chunk_paths)}")
print(f"Test chunks : {len(test_chunk_paths)}")

pos_weight_value = np.sum(tv_labels==0) / np.sum(tv_labels==1)
print(f"pos_weight  : {pos_weight_value:.4f}")

# ── Load chunks into RAM ─────────────────────────────────────
print("\nLoading TV chunks into RAM...")
all_tv_mels, all_tv_mfccs = [], []
for path in tqdm(tv_chunk_paths):
    chunk = np.load(path)
    all_tv_mels.append(chunk['mels'])
    all_tv_mfccs.append(chunk['mfccs'])
    del chunk; gc.collect()
all_tv_mels  = np.concatenate(all_tv_mels,  axis=0)
all_tv_mfccs = np.concatenate(all_tv_mfccs, axis=0)
print(f"TV mels  : {all_tv_mels.shape}")
print(f"TV mfccs : {all_tv_mfccs.shape}")

print("\nLoading Test chunks into RAM...")
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

# ── Dataset ──────────────────────────────────────────────────
class ParkinsonsDataset(Dataset):
    def __init__(self, mels, mfccs, labels, augment=False):
        self.mels    = mels
        self.mfccs   = mfccs
        self.labels  = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _time_mask(self, mel, max_size=20):
        mel = mel.copy()
        t0  = random.randint(0, mel.shape[1]-max_size)
        mel[:, t0:t0+max_size] = 0.0
        return mel

    def _freq_mask(self, mel, max_size=20):
        mel = mel.copy()
        f0  = random.randint(0, mel.shape[0]-max_size)
        mel[f0:f0+max_size, :] = 0.0
        return mel

    def __getitem__(self, idx):
        mel  = self.mels[idx].copy()
        mfcc = self.mfccs[idx].copy()
        lbl  = float(self.labels[idx])
        if self.augment:
            if random.random() < 0.5:
                mel = self._time_mask(mel)
            if random.random() < 0.5:
                mel = self._freq_mask(mel)
        mel_tensor  = torch.tensor(mel,  dtype=torch.float32).unsqueeze(0)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        lbl_tensor  = torch.tensor(lbl,  dtype=torch.float32)
        return mel_tensor, mfcc_tensor, lbl_tensor

# ── Model ─────────────────────────────────────────────────────
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
        self._printed = False

    def forward(self, mel, mfcc, ast_in):
        c = self.cnn_branch(mel)
        b = self.bilstm_branch(mfcc)
        a = self.ast_branch(ast_in)
        if not self._printed:
            print(f"  CNN:{c.shape} BiLSTM:{b.shape} AST:{a.shape}")
            self._printed = True
        fused   = torch.stack([c,b,a], dim=1)
        attn, _ = self.attention(fused, fused, fused)
        return self.classifier(attn.reshape(attn.size(0), -1))


def prepare_ast_input(mels_batch):
    mels = mels_batch.squeeze(1)
    ast_input = torch.nn.functional.interpolate(
        mels.unsqueeze(1),
        size=(1024, 128),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)
    return ast_input

# ── Training utilities ────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for mel, mfcc, lbl in loader:
        mel  = mel.to(device)
        mfcc = mfcc.to(device)
        lbl  = lbl.to(device)
        ast_input = prepare_ast_input(mel)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out  = model(mel, mfcc, ast_input).squeeze(1)
            loss = criterion(out, lbl)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds = torch.sigmoid(out).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbl.cpu().numpy())
    return total_loss/len(loader), roc_auc_score(all_labels, all_preds)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, mfcc, lbl in loader:
            mel  = mel.to(device)
            mfcc = mfcc.to(device)
            lbl  = lbl.to(device)
            ast_input = prepare_ast_input(mel)
            with torch.amp.autocast('cuda'):
                out  = model(mel, mfcc, ast_input).squeeze(1)
                loss = criterion(out, lbl)
            total_loss += loss.item()
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.cpu().numpy())
    avg_loss = total_loss/len(loader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, [1 if p>0.5 else 0 for p in all_preds])
    f1  = f1_score(all_labels, [1 if p>0.5 else 0 for p in all_preds])
    return avg_loss, auc, acc, f1, all_preds, all_labels

# ── Training — folds 2 and 3 only ────────────────────────────
gkf = GroupKFold(n_splits=N_FOLDS)
fold_results = []

all_splits = list(gkf.split(all_tv_mels, tv_labels, tv_parents))
valid_splits = []
for fold, (train_idx, val_idx) in enumerate(all_splits):
    val_lbl = tv_labels[val_idx]
    hc = np.sum(val_lbl==0)
    pd = np.sum(val_lbl==1)
    print(f"Fold {fold+1} — Val HC:{hc} PD:{pd} "
          f"{'✅' if hc>0 and pd>0 else '❌ skipping'}")
    if hc > 0 and pd > 0:
        valid_splits.append((fold+1, train_idx, val_idx))

print(f"\nTraining {len(valid_splits)} valid folds")

pos_weight = torch.tensor([pos_weight_value],
                           dtype=torch.float32).to(device)

for fold_num, train_idx, val_idx in valid_splits:
    print(f"\n{'='*55}")
    print(f"  FOLD {fold_num}")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")
    val_lbl = tv_labels[val_idx]
    print(f"  Val HC: {np.sum(val_lbl==0)} PD: {np.sum(val_lbl==1)}")
    print(f"{'='*55}")

    train_ds = ParkinsonsDataset(
        all_tv_mels[train_idx], all_tv_mfccs[train_idx],
        tv_labels[train_idx], augment=True)
    val_ds = ParkinsonsDataset(
        all_tv_mels[val_idx], all_tv_mfccs[val_idx],
        tv_labels[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = HybridPDDetector().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda')

    best_auc    = 0.0
    patience_ct = 0
    history     = {'train_loss':[], 'val_loss':[],
                   'train_auc':[], 'val_auc':[]}

    for epoch in range(MAX_EPOCHS):
        tr_loss, tr_auc = train_epoch(
            model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_auc, vl_acc, vl_f1, _, _ = eval_epoch(
            model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_auc'].append(tr_auc)
        history['val_auc'].append(vl_auc)

        print(f"  Ep {epoch+1:02d}/{MAX_EPOCHS} | "
              f"TrLoss:{tr_loss:.4f} TrAUC:{tr_auc:.4f} | "
              f"VlLoss:{vl_loss:.4f} VlAUC:{vl_auc:.4f} "
              f"VlAcc:{vl_acc:.4f}")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(),
                os.path.join(MODEL_DIR, f'fold{fold_num}_best.pth'))
            print(f"  ✅ Saved fold{fold_num}_best.pth "
                  f"(AUC: {best_auc:.4f})")
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    fold_results.append({
        'fold'    : fold_num,
        'best_auc': best_auc,
        'history' : history
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title(f'Fold {fold_num} — Loss')
    axes[0].set_xlabel('Epoch'); axes[0].legend()
    axes[1].plot(history['train_auc'], label='Train')
    axes[1].plot(history['val_auc'],   label='Val')
    axes[1].set_title(f'Fold {fold_num} — AUC')
    axes[1].set_xlabel('Epoch'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'fold{fold_num}_curves.png'),
                dpi=150)
    plt.close()

    del model, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

# Summary
print(f"\n{'='*55}")
print(f"  CROSS-VALIDATION SUMMARY")
print(f"{'='*55}")
aucs = [r['best_auc'] for r in fold_results]
print(f"  {'Fold':<8} {'Val AUC'}")
for r in fold_results:
    print(f"  {r['fold']:<8} {r['best_auc']:.4f}")
print(f"  {'─'*20}")
print(f"  Mean     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"{'='*55}")
print("\n✅ Training complete!")