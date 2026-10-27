import torch


def _partially_linear(
    inputs: torch.Tensor,           # [BATCH, N_EDITED, C_IN]
    proj: torch.nn.Module,          # to_k / to_v
    index: torch.Tensor,            # positions in the full cache
    outputs: torch.Tensor,          # [BATCH, N_FULL, C_OUT]
) -> None:
    """
    Project edited tokens and write them into the cached K/V rows.

    The original Triton kernel multiplied ``proj.weight`` directly. Quantized
    linears (bitsandbytes, Comfy, GGUF) store packed integer weights, so that
    matmul fails. Calling the module uses the real quantized forward and LoRA.
    """
    updated = proj(inputs)
    if updated.ndim == 2:
        updated = updated.unsqueeze(0)
    if updated.shape[-1] != outputs.shape[-1]:
        updated = updated.reshape(outputs.shape[0], updated.shape[1], outputs.shape[-1])
    if updated.dtype != outputs.dtype:
        updated = updated.to(dtype=outputs.dtype)
    index = index.to(device=outputs.device, dtype=torch.long)
    outputs[:, index] = updated
