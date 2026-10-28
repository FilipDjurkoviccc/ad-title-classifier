"""Treniranje MLP mreže epohu po epohu (da bi se mogle pratiti krive učenja)."""
import copy
import time

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier


def train_mlp(Xtr, ytr, Xva, yva, hidden=(64,), alpha=1e-4, lr=1e-3, batch_size=128,
              max_epochs=25, patience=3, seed=0, verbose=False):
    """Vraća (model sa najboljim težinama po validacionom gubitku, istorija)."""
    classes = np.unique(ytr)
    model = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=lr,
                          batch_size=batch_size, random_state=seed)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_model, bad = np.inf, None, 0
    for ep in range(max_epochs):
        t0 = time.time()
        model.partial_fit(Xtr, ytr, classes=classes)
        ptr, pva = model.predict_proba(Xtr), model.predict_proba(Xva)
        hist["train_loss"].append(log_loss(ytr, ptr, labels=classes))
        hist["val_loss"].append(log_loss(yva, pva, labels=classes))
        hist["train_acc"].append(accuracy_score(ytr, ptr.argmax(1)))
        hist["val_acc"].append(accuracy_score(yva, pva.argmax(1)))
        if verbose:
            print(f"epoha {ep+1:2d} | train loss {hist['train_loss'][-1]:.4f} acc {hist['train_acc'][-1]:.4f}"
                  f" | val loss {hist['val_loss'][-1]:.4f} acc {hist['val_acc'][-1]:.4f} | {time.time()-t0:.0f}s")
        if hist["val_loss"][-1] < best - 1e-4:
            best, best_model, bad = hist["val_loss"][-1], copy.deepcopy(model), 0
        else:
            bad += 1
            if bad >= patience:
                break
    hist["best_epoch"] = int(np.argmin(hist["val_loss"])) + 1
    return best_model, hist

