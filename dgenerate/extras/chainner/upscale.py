import math
import PIL.Image
import numpy as np
import torch
from tqdm.auto import tqdm
from .types import PyTorchModel


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


@torch.inference_mode()
def tiled_scale(samples: torch.Tensor, upscale_model: PyTorchModel, tile_x=64, tile_y=64, overlap=8, upscale_amount=4,
                out_channels=3, pbar=None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount),
                          round(samples.shape[3] * upscale_amount)), device="cpu")
    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device="cpu")
        out_div = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device="cpu")
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y:y + tile_y, x:x + tile_x]

                ps = upscale_model(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t:1 + t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, mask.shape[2] - 1 - t: mask.shape[2] - t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, t:1 + t] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, mask.shape[3] - 1 - t: mask.shape[3] - t] *= ((1.0 / feather) * (t + 1))
                out[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += ps * mask
                out_div[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)
        output[b:b + 1] = out / out_div
    return output


def upscale(upscale_model: PyTorchModel, image, tile=512, overlap=32, device='cuda'):
    image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

    upscale_model.to(device)

    in_img = image.movedim(-1, -3).to(device)

    oom = True

    while oom:
        try:
            steps = in_img.shape[0] * get_tiled_scale_steps(
                in_img.shape[3],
                in_img.shape[2],
                tile_x=tile,
                tile_y=tile,
                overlap=overlap)

            pbar = tqdm(total=steps)

            s = tiled_scale(in_img,
                            upscale_model,
                            tile_x=tile,
                            tile_y=tile,
                            overlap=overlap,
                            upscale_amount=upscale_model.scale,
                            pbar=pbar)

            oom = False
        except torch.cuda.OutOfMemoryError as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.cpu()

    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

    return PIL.Image.fromarray(np.clip(255. * s[0].cpu().numpy(), 0, 255).astype(np.uint8))
