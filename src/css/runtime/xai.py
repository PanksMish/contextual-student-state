"""Explainability: Grad-CAM overlays and deletion curves for the facial branch (paper Section VIII-A).
Attribution supports inspection only and does not validate a psychological interpretation."""
from __future__ import annotations
import numpy as np


def grad_cam(model, layer, x, class_idx=None, reshape=None):
    """x: (1,3,H,W) tensor. `layer` is a conv block (ResNet layer4) or a Swin stage (pass reshape to NHWC->NCHW)."""
    import torch
    acts, grads = [], []
    h1 = layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    out = model(x); c = int(out.argmax(1)) if class_idx is None else class_idx
    model.zero_grad(); out[0, c].backward(); h1.remove(); h2.remove()
    a, g = acts[0], grads[0]
    if reshape: a, g = reshape(a), reshape(g)
    w = g.mean(dim=(2, 3), keepdim=True); cam = torch.relu((w * a).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-9), c


def deletion_curve(model, x, cam, class_idx, fractions=(0.0, 0.1, 0.25, 0.5), fill=0.0):
    """Probability of the predicted class after removing the most salient pixels (mean fill)."""
    import torch
    order = np.argsort(-cam.ravel()); res = []
    for f in fractions:
        xm = x.clone(); k = int(f * cam.size)
        if k:
            m = np.zeros(cam.size, bool); m[order[:k]] = True
            xm[0, :, torch.from_numpy(m.reshape(cam.shape))] = fill
        with torch.no_grad():
            res.append(float(torch.softmax(model(xm), -1)[0, class_idx]))
    return res
