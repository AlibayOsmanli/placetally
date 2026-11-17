"""Per-frame features and the neighbouring-frame features the cut detector reads."""

import numpy as np
from PIL import Image

# Degrees to rotate a frame upright, from the recording's orientation label.
ROTATION = {"Up": 0, "Left": -90, "Down": 180, "Right": 90}


def unit(x):
    """Rows scaled to unit length, in float32."""
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def upright(image, orientation):
    angle = ROTATION.get(orientation, 0)
    return image.rotate(angle, expand=True) if angle else image


def frame_features(path, orientation):
    """Return the 384x384 encoder input, the grey thumbnail and the colour histogram of one frame."""
    im = upright(Image.open(path).convert("RGB"), orientation)
    encoder_input = np.asarray(im.resize((384, 384)))

    side = min(im.width, im.height)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    square = im.crop((left, top, left + side, top + side))

    thumb = np.asarray(square.convert("L").resize((28, 28), Image.BILINEAR), np.float32)
    thumb = thumb.reshape(-1) - thumb.mean()
    thumb = thumb / max(np.linalg.norm(thumb), 1e-6)

    small = np.asarray(square.resize((84, 84), Image.BILINEAR))
    q = (small // 32).reshape(-1, 3).astype(np.int64)
    hist = np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512).astype(np.float32)
    return encoder_input, thumb, hist / hist.sum()


class FrameEncoder:
    """SigLIP 2 image codes (the pooled output, L2-normalised)."""

    def __init__(self, checkpoint, device="cuda"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.device = device

    def encode(self, images, batch=64):
        torch = self.torch
        codes = []
        with torch.inference_mode():
            for start in range(0, len(images), batch):
                pixels = self.processor(images=list(images[start : start + batch]), return_tensors="pt")["pixel_values"]
                out = self.model.vision_model(pixel_values=pixels.to(self.device, torch.bfloat16))
                codes.append(torch.nn.functional.normalize(out.pooler_output.float(), dim=-1).half().cpu())
        return torch.cat(codes).numpy()


def pair_features(thumbnail, histogram, code):
    """Six features for each pair of neighbouring frames.

    Three similarities (thumbnail correlation, histogram intersection, code cosine), then each one minus
    its median over the five pairs on either side.
    """
    sims = np.stack(
        [
            (thumbnail[1:] * thumbnail[:-1]).sum(1),
            np.minimum(histogram[1:], histogram[:-1]).sum(1),
            (code[1:] * code[:-1]).sum(1),
        ],
        1,
    )
    n = len(sims)
    relative = np.zeros_like(sims)
    for i in range(n):
        around = np.concatenate([sims[max(0, i - 5) : i], sims[i + 1 : min(n, i + 6)]])
        relative[i] = sims[i] - (np.median(around, 0) if len(around) else sims[i])
    return np.concatenate([sims, relative], 1)
