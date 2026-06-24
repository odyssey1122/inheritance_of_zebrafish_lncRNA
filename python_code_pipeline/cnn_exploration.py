import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
import sys

# --- THE "OVERFIT-KILLER" ARCHITECTURE ---
class FinalMScCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super(FinalMScCNN, self).__init__()
        # Parallel scanning windows
        self.conv8 = nn.Conv1d(4, 16, kernel_size=8, padding=4)
        self.conv15 = nn.Conv1d(4, 16, kernel_size=15, padding=7)
        self.conv20 = nn.Conv1d(4, 16, kernel_size=20, padding=10)
        
        # Batch Norm for each branch separately is more stable
        self.bn8 = nn.BatchNorm1d(16)
        self.bn15 = nn.BatchNorm1d(16)
        self.bn20 = nn.BatchNorm1d(16)
        
        # We have 3 branches, each produces a GAP and a GMP score (16+16 each)
        # Total input features = (16+16) * 3 = 96
        self.fc1 = nn.Linear(96, 32) 
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Scan with different scales
        out8 = F.relu(self.bn8(self.conv8(x)))
        out15 = F.relu(self.bn15(self.conv15(x)))
        out20 = F.relu(self.bn20(self.conv20(x)))
        
        # 2. POOL SEPARATELY (This prevents the RuntimeError)
        # We take the average (GAP) and the max (GMP) for each kernel scale
        gap8, gmp8 = out8.mean(dim=2), out8.max(dim=2).values
        gap15, gmp15 = out15.mean(dim=2), out15.max(dim=2).values
        gap20, gmp20 = out20.mean(dim=2), out20.max(dim=2).values
        
        # 3. Combine all pooled features (Total size: 96)
        combined = torch.cat([gap8, gmp8, gap15, gmp15, gap20, gmp20], dim=1)
        
        # 4. Final Dense layers
        x = self.dropout(F.relu(self.fc1(combined)))
        return self.fc2(x)


# --- FASTA DATA LOADER ---
class FastaDataset(Dataset):
    def __init__(self, inh_path, non_inh_path):
        self.samples = []
        self._load(inh_path, 1)
        self._load(non_inh_path, 0)
        random.shuffle(self.samples)

    def _load(self, path, label):
        print(f"Loading {path}...")
        try:
            with open(path, 'r') as f:
                name, seq = None, []
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if seq: self._process("".join(seq), label)
                        seq = []
                    elif line: seq.append(line)
                if seq: self._process("".join(seq), label)
        except FileNotFoundError:
            print(f"CRITICAL ERROR: {path} not found.")
            sys.exit(1)

    def _process(self, seq, label):
        mapping = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'U':[0,0,0,1],'T':[0,0,0,1]}
        encoded = [mapping.get(b.upper(), [0,0,0,0]) for b in seq]
        if len(encoded) > 25: # Filter out short noise
            tensor = torch.FloatTensor(encoded).transpose(0, 1)
            self.samples.append((tensor, torch.FloatTensor([label])))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# --- TRAINING ENGINE ---
def main():
    # 1. Prepare Data
    ds = FastaDataset("inherited_train.fasta", "non_inherited_train.fasta")
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    # Calculate Class Weights to stop the 52% flatline
    n_inh = sum([1 for _, y in ds if y.item() == 1])
    n_non = len(ds) - n_inh
    # Weight = ratio of Non-Inherited to Inherited
    p_weight = torch.tensor([n_non / n_inh])
    print(f"Class Weight (Inherited): {p_weight.item():.2f}")

    # 2. Setup "The Brain"
    model = FinalMScCNN(dropout=0.3)
    # Weight decay (1e-4) is the key to stopping overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=p_weight)

    print(f"\nTraining on {train_size} seqs | Validating on {val_size} seqs")
    print("-" * 40)

    best_val_acc = 0
    
    for epoch in range(20):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # Validation Check
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = (torch.sigmoid(model(x)) > 0.5).int()
                if pred.item() == int(y.item()): correct += 1
        
        val_acc = (correct / val_size) * 100
        print(f"Epoch {epoch+1:02d} | Train Loss: {t_loss/train_size:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc: best_val_acc = val_acc

    print("-" * 40)
    print(f"STUDY COMPLETE. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
