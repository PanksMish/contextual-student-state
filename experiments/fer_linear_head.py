"""Frozen-backbone transfer on facial expressions (Sections V-C and V-D).

Frozen ImageNet backbones (ResNet18, ResNet50, Swin-T) see 128x128 RGB inputs with ImageNet
normalisation. Features are extracted once (no augmentation, so freezing makes this exact),
and a newly initialised 7-class linear head is trained for 20 epochs with class-weighted
cross-entropy, AdamW (lr 0.001, weight decay 0.01) and batches of 128. Validation macro F1
selects the epoch. Head sizes: ResNet18 3,591; Swin-T 5,383; ResNet50 14,343 parameters.

usage: python -m experiments.fer_linear_head DATA.npz OUT.json [--seeds 9301 9302 9303]
DATA.npz holds x (N,H,W[,3]) uint8, y (N,), part in {train,val,test} (use fer_splits to build it
from the manifest, or the official partition with train / public_test / private_test).
"""
from __future__ import annotations
import argparse
import numpy as np
from .common import set_seed, macro_f1, accuracy, save_json

BACKBONES = ("resnet18", "resnet50", "swin_t")


def build_backbone(name: str):
    import torch, torchvision
    from torchvision import models
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); m.fc = torch.nn.Identity(); d = 512
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1); m.fc = torch.nn.Identity(); d = 2048
    elif name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1); m.head = torch.nn.Identity(); d = 768
    else:
        raise ValueError(name)
    return m.eval(), d


@__import__("torch").no_grad()
def extract(name: str, x: np.ndarray, device: str = "cpu", batch: int = 128) -> np.ndarray:
    import torch, torch.nn.functional as F
    model, _ = build_backbone(name); model.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(x), batch):
        b = torch.from_numpy(x[i:i + batch]).to(device).float() / 255.0
        b = b.unsqueeze(1).repeat(1, 3, 1, 1) if b.ndim == 3 else b.permute(0, 3, 1, 2)
        b = F.interpolate(b, size=(128, 128), mode="bilinear", align_corners=False)
        out.append(model((b - mean) / std).cpu().numpy())
    return np.concatenate(out)


def train_head(ftr, ytr, fva, yva, fte, yte, seed: int, n_classes: int = 7, epochs: int = 20, device: str = "cpu"):
    import torch
    set_seed(seed)
    head = torch.nn.Linear(ftr.shape[1], n_classes).to(device)
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    Xtr, Ytr = torch.from_numpy(ftr).float().to(device), torch.from_numpy(ytr).long().to(device)
    pred = lambda f: head(torch.from_numpy(f).float().to(device)).argmax(1).cpu().numpy()
    best, best_state = -1.0, None
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); lossf(head(Xtr[idx]), Ytr[idx]).backward(); opt.step()
        head.eval()
        f1 = macro_f1(yva, pred(fva))
        if f1 > best:
            best = f1; best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    p = pred(fte)
    return dict(val_macro_f1=best, test_acc=accuracy(yte, p), test_macro_f1=macro_f1(yte, p), pred=p.tolist(),
                n_params=sum(q.numel() for q in head.parameters()))


def paired_image_bootstrap(correct_a, correct_b, n_boot=5000, seed=0):
    """Image-level bootstrap of the accuracy difference (conditions on test images, ignores identity)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(correct_a, float) - np.asarray(correct_b, float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[9301, 9302, 9303])
    ap.add_argument("--test-part", default="test"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    z = np.load(a.data, allow_pickle=True); x, y, part = z["x"], z["y"].astype(int), z["part"]
    sel = {"train": part == "train", "val": part == ("val" if "val" in part else "public_test"), "test": part == a.test_part}
    res = {}
    for name in BACKBONES:
        feats = extract(name, x, a.device)
        res[name] = [train_head(feats[sel["train"]], y[sel["train"]], feats[sel["val"]], y[sel["val"]],
                                feats[sel["test"]], y[sel["test"]], s, device=a.device) for s in a.seeds]
        accs = [r["test_acc"] for r in res[name]]
        print(name, "acc %.4f +- %.4f" % (np.mean(accs), np.std(accs, ddof=1)))
    yte = y[sel["test"]]
    ca = (np.array(res["swin_t"][0]["pred"]) == yte); cb = (np.array(res["resnet50"][0]["pred"]) == yte)
    res["swin_minus_resnet50_acc_seed0"] = paired_image_bootstrap(ca, cb)
    save_json(res, a.out)
