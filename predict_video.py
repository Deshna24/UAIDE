import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from video_data import read_video_frames, FaceCropper
from video_model import ResNetLSTM, GradCAM, overlay_cam


def build_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model = ResNetLSTM(
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 1),
        bidirectional=config.get("bidirectional", True),
        temporal_pool=config.get("temporal_pool", "mean"),
        pretrained=config.get("pretrained", False),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="video_resnet_lstm.pt")
    parser.add_argument("--frames_per_video", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=None)
    parser.add_argument("--no_face_detection", action="store_true")
    parser.add_argument("--gradcam_out", type=str, default="gradcam_frames")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint_path = Path(args.checkpoint)
    model, config = load_model(checkpoint_path, device)

    frames_per_video = args.frames_per_video or config.get("frames_per_video", 16)
    frame_stride = args.frame_stride or config.get("frame_stride", 4)
    face_detection = config.get("face_detection", True)
    if args.no_face_detection:
        face_detection = False

    face_cropper = FaceCropper() if face_detection else None
    frames = read_video_frames(
        Path(args.video),
        frames_per_video=frames_per_video,
        frame_stride=frame_stride,
        face_cropper=face_cropper,
    )

    transform = build_transforms()
    video_tensor = torch.stack([transform(Image.fromarray(f)) for f in frames], dim=0)
    video_tensor = video_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        frame_logits, video_logits = model(video_tensor)
        video_probs = torch.softmax(video_logits, dim=1).cpu().numpy()[0]
        frame_probs = torch.softmax(frame_logits.squeeze(0), dim=1).cpu().numpy()

    pred = int(np.argmax(video_probs))
    label = "fake" if pred == 1 else "real"
    print(f"Prediction: {label} | prob_real={video_probs[0]:.4f} prob_fake={video_probs[1]:.4f}")

    # Grad-CAM for top-k frames by fake probability
    out_dir = Path(args.gradcam_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_k = min(args.top_k, len(frames))
    top_idx = np.argsort(frame_probs[:, 1])[::-1][:top_k]

    target_layer = model.backbone.layer4[-1].conv3
    cam = GradCAM(model, target_layer)

    for rank, idx in enumerate(top_idx, start=1):
        frame = frames[idx]
        frame_tensor = transform(Image.fromarray(frame)).unsqueeze(0).unsqueeze(0).to(device)
        cam_map = cam.generate(frame_tensor, class_idx=pred)
        overlay = overlay_cam(frame, cam_map, alpha=0.5)
        out_path = out_dir / f"frame_{idx:04d}_rank{rank}.png"
        save_image(overlay, out_path)
def save_image(arr, path):
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    main()
