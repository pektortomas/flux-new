"""Run before downloading gigabytes: a real CUDA operation, not just GPU enumeration."""
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Deploy this image to an NVIDIA GPU Pod, not a CPU Pod.")
gpu = torch.cuda.get_device_properties(0)
capability = torch.cuda.get_device_capability(0)
cuda = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
if capability >= (12, 0) and cuda < (12, 8):
    raise SystemExit("Blackwell / RTX 5090 requires a CUDA 12.8+ PyTorch build.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("This configuration requires native BF16 support (Ampere or newer).")
x = torch.ones((256, 256), device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
print(json.dumps({"gpu": gpu.name, "vram_gib": round(gpu.total_memory / 1024**3, 1),
                  "compute_capability": capability, "torch": torch.__version__,
                  "cuda": torch.version.cuda, "cuda_matmul": "passed"}))
del x
torch.cuda.empty_cache()
