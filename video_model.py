import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models


class ResNetLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        temporal_pool: str = "mean",
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.temporal_pool = temporal_pool

        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        except Exception:
            backbone = models.resnet50(pretrained=pretrained)

        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.frame_head = nn.Linear(2048, num_classes)

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.video_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, t, -1)

        frame_logits = self.frame_head(feats)

        lstm_out, _ = self.lstm(feats)
        if self.temporal_pool == "last":
            pooled = lstm_out[:, -1, :]
        else:
            pooled = lstm_out.mean(dim=1)
        video_logits = self.video_head(pooled)
        return frame_logits, video_logits


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(self._save_gradient)
        else:
            self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        # input_tensor: (1, 1, C, H, W)
        self.model.eval()
        frame_logits, _ = self.model(input_tensor)
        scores = frame_logits.squeeze(1)
        target = scores[:, class_idx]

        self.model.zero_grad()
        target.backward(retain_graph=True)

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-12)
        return cam


def overlay_cam(frame_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    cam_resized = cv2.resize(cam, (frame_rgb.shape[1], frame_rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay
