"""Face branch: detect a face, run a frozen backbone with a linear head, keep only the 7 expression
probabilities and a quality score. The frame is never stored (paper Section IX)."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from ..state import expression_index, quality_face

CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")


class FaceExpressionEncoder:
    def __init__(self, head_path: str, backbone: str = "swin_t", device: str = "cpu"):
        import torch
        from torchvision import models
        self.torch, self.device = torch, device
        if backbone == "swin_t":
            self.net = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1); self.net.head = torch.nn.Identity(); d = 768
        elif backbone == "resnet50":
            self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1); self.net.fc = torch.nn.Identity(); d = 2048
        else:
            self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); self.net.fc = torch.nn.Identity(); d = 512
        self.net.eval().to(device)
        self.head = torch.nn.Linear(d, 7); self.head.load_state_dict(torch.load(head_path, map_location=device)); self.head.eval().to(device)
        import cv2
        self.cv2 = cv2; self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[Optional[Dict[str, float]], float]:
        cv2, torch = self.cv2, self.torch
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None, 0.0
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = cv2.resize(frame_bgr[y:y + h, x:x + w], (128, 128))
        sharp = min(1.0, cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() / 100.0)
        size = min(1.0, (w * h) / (0.1 * frame_bgr.shape[0] * frame_bgr.shape[1]))
        t = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            p = torch.softmax(self.head(self.net(((t - self.mean) / self.std).to(self.device))), -1)[0].cpu().numpy()
        return dict(zip(CLASSES, map(float, p))), quality_face(1.0, sharp * size)


def expression_summary(probs: Optional[Dict[str, float]]) -> Optional[float]:
    return None if probs is None else expression_index(probs)
