"""Transformer klasifikator naslova (PyTorch, trenira se od nule, bez pretreniranih modela)."""
import copy
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = 0, 1


def build_vocab(texts, min_count=2):
    """Rečnik: svaka reč koja se u treningu pojavi bar min_count puta dobija svoj broj."""
    counts = Counter(w for t in texts for w in t.split())
    words = sorted(w for w, n in counts.items() if n >= min_count)
    return {w: i + 2 for i, w in enumerate(words)}   # 0 = PAD, 1 = UNK


def encode(texts, vocab, max_len=14):
    """Naslov -> niz brojeva fiksne dužine (kraći se dopunjavaju sa PAD)."""
    X = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, UNK) for w in t.split()][:max_len]
        X[i, :len(ids)] = ids
    return torch.tensor(X)


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, n_classes=10, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, max_len=14, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)   
        self.pos = nn.Embedding(max_len, d_model)                     
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, x):
        pad = x == PAD
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        h = self.encoder(h, src_key_padding_mask=pad)
        
        h = h.masked_fill(pad.unsqueeze(-1), 0).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        return self.out(h)


def evaluate(model, X, y, bs=2048):
    """Vraća (gubitak, tačnost, verovatnoće) bez menjanja težina."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc, torch.softmax(logits, 1).numpy()


def train(Xtr, ytr, Xva, yva, vocab_size, lr=1e-3, weight_decay=0.01, batch_size=128,
          max_epochs=30, patience=3, seed=0, verbose=False, **arch):
    """Trenira epohu po epohu, vraća (model sa najboljim težinama po validacionom gubitku, istorija)."""
    torch.manual_seed(seed)
    model = TextTransformer(vocab_size, **arch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        t0 = time.time()
        model.train()
        order = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = order[i:i + batch_size]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()    
            opt.step()
        tl, ta, _ = evaluate(model, Xtr, ytr)
        vl, va, _ = evaluate(model, Xva, yva)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        if verbose:
            print(f"epoha {ep+1:2d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f} | {time.time()-t0:.0f}s")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["val_loss"])) + 1
    return model, hist
