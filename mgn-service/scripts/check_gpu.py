import torch
import sys

print("=" * 60)
print("PyTorch GPU Check")
print("=" * 60)
print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA:    {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} | {props.total_memory / 1024**3:.1f} GB VRAM")
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    _ = a @ b
    print("✅ GPU test passed")
else:
    print("CUDA is not available — training will run on CPU.")
    print("For GPU install a CUDA build of PyTorch from pytorch.org")
print("=" * 60)
