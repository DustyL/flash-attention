# B300/SM103 Flash-Attention Testing Guide

**Branch:** `b300-sm103-optimization`
**Target GPU:** NVIDIA B300 (SM 10.3, Blackwell)
**Last Updated:** 2026-02-05

---

## Quick Start

```bash
# Clone and checkout the optimization branch
git clone https://github.com/DustyL/flash-attention.git
cd flash-attention
git checkout b300-sm103-optimization

# Build for SM103 only (fastest)
export FLASH_ATTN_CUDA_ARCHS="103"
export MAX_JOBS=8
export NVCC_THREADS=4
pip wheel . --no-build-isolation --no-deps -v 2>&1 | tee build.log

# Install the wheel
pip install --force-reinstall --no-deps flash_attn-*.whl
```

---

## 1. Build Verification

### 1.1 Verify SM103 Gencode Flags
```bash
# Check build log for correct gencode
grep -E "compute_103|sm_103" build.log

# Expected (CUDA 13.0+):
#   -gencode arch=compute_103f,code=sm_103
# Expected (CUDA 12.8-12.9):
#   -gencode arch=compute_103,code=sm_103
```

### 1.2 Verify Native Kernels Compiled
```bash
python3 -c "
import flash_attn_2_cuda
import subprocess
import shutil

so_path = flash_attn_2_cuda.__file__
cuobjdump = shutil.which('cuobjdump')
if cuobjdump:
    out = subprocess.check_output([cuobjdump, '--list-elf', so_path], text=True)
    sm103_count = out.count('sm_103')
    print(f'SM103 kernels: {sm103_count}')
    print('Status:', 'PASS' if sm103_count >= 70 else 'FAIL (expected 70+)')
else:
    print('cuobjdump not found - install CUDA toolkit')
"
```

### 1.3 Verify Version
```bash
python3 -c "import flash_attn; print(f'Version: {flash_attn.__version__}')"
# Expected: 2.8.3 or similar
```

---

## 2. Functional Tests

### 2.1 Basic Forward Pass
```python
import torch
from flash_attn import flash_attn_func

# Setup
device = 'cuda'
dtype = torch.bfloat16
batch, seqlen, heads, headdim = 2, 2048, 16, 128

q = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype)
k = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype)
v = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype)

# Forward
out = flash_attn_func(q, k, v)
print(f"Forward: PASS - shape={out.shape}, dtype={out.dtype}")
```

### 2.2 Forward + Backward Pass
```python
import torch
from flash_attn import flash_attn_func

device = 'cuda'
dtype = torch.bfloat16
batch, seqlen, heads, headdim = 2, 2048, 16, 128

q = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)

# Forward + Backward
out = flash_attn_func(q, k, v)
loss = out.sum()
loss.backward()

print(f"Backward: {'PASS' if q.grad is not None else 'FAIL'}")
print(f"  q.grad shape: {q.grad.shape}")
print(f"  k.grad shape: {k.grad.shape}")
print(f"  v.grad shape: {v.grad.shape}")
```

### 2.3 Variable-Length Sequences (Varlen)
```python
import torch
from flash_attn import flash_attn_varlen_func

device = 'cuda'
dtype = torch.bfloat16

# Two sequences: 512 and 1024 tokens
cu_seqlens = torch.tensor([0, 512, 1536], dtype=torch.int32, device=device)
total_tokens = 1536
heads, headdim = 16, 128
max_seqlen = 1024

q = torch.randn(total_tokens, heads, headdim, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(total_tokens, heads, headdim, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(total_tokens, heads, headdim, device=device, dtype=dtype, requires_grad=True)

# Forward + Backward
out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
out.sum().backward()

print(f"Varlen Forward: PASS - shape={out.shape}")
print(f"Varlen Backward: {'PASS' if q.grad is not None else 'FAIL'}")
```

### 2.4 Causal Attention
```python
import torch
from flash_attn import flash_attn_func

device = 'cuda'
dtype = torch.bfloat16
batch, seqlen, heads, headdim = 2, 2048, 16, 128

q = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)

out = flash_attn_func(q, k, v, causal=True)
out.sum().backward()

print(f"Causal Forward: PASS - shape={out.shape}")
print(f"Causal Backward: {'PASS' if q.grad is not None else 'FAIL'}")
```

---

## 3. CuTE Interface Tests

CuTE (CUDA Templates) provides the optimized Blackwell kernels.

### 3.1 Install CuTE Dependencies
```bash
pip install 'quack-kernels==0.2.4'
```

### 3.2 Test CuTE Import
```python
try:
    from flash_attn.cute import flash_attn_func as cute_flash_attn_func
    print("CuTE Import: PASS")
except ImportError as e:
    print(f"CuTE Import: FAIL - {e}")
    print("Run: pip install 'quack-kernels==0.2.4'")
```

### 3.3 CuTE Forward + Backward
```python
import torch
from flash_attn.cute import flash_attn_func

device = 'cuda'
dtype = torch.bfloat16
batch, seqlen, heads, headdim = 2, 2048, 16, 128

q = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)

# CuTE forward (returns tuple: out, lse)
out, lse = flash_attn_func(q, k, v)
out.sum().backward()

print(f"CuTE Forward: PASS - shape={out.shape}")
print(f"CuTE Backward: {'PASS' if q.grad is not None else 'FAIL'}")
print(f"CuTE LSE shape: {lse.shape}")
```

---

## 4. Performance Benchmarks

### 4.1 Quick Benchmark
```python
import torch
import time
from flash_attn import flash_attn_func

device = 'cuda'
dtype = torch.bfloat16

def benchmark(batch, seqlen, heads, headdim, warmup=5, iters=20):
    q = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)

    # Warmup
    for _ in range(warmup):
        out = flash_attn_func(q, k, v)
        out.sum().backward()
        q.grad = k.grad = v.grad = None

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        out = flash_attn_func(q, k, v)
        out.sum().backward()
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    end = time.perf_counter()

    ms_per_iter = (end - start) / iters * 1000
    return ms_per_iter

# Run benchmarks
configs = [
    (2, 1024, 16, 128),
    (2, 2048, 16, 128),
    (2, 4096, 16, 128),
    (4, 2048, 16, 128),
]

print("B300 SM103 Flash-Attention Benchmark")
print("=" * 50)
print(f"{'Config':<25} {'Time (ms)':<15}")
print("-" * 50)

for batch, seqlen, heads, headdim in configs:
    ms = benchmark(batch, seqlen, heads, headdim)
    config_str = f"B={batch}, L={seqlen}, H={heads}, D={headdim}"
    print(f"{config_str:<25} {ms:.3f}")
```

### 4.2 Compare with PyTorch SDPA
```python
import torch
import torch.nn.functional as F
import time

device = 'cuda'
dtype = torch.bfloat16

def benchmark_comparison(batch, seqlen, heads, headdim, warmup=5, iters=20):
    from flash_attn import flash_attn_func

    # Flash-Attention uses [B, L, H, D]
    q_fa = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
    k_fa = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)
    v_fa = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dtype, requires_grad=True)

    # PyTorch SDPA uses [B, H, L, D]
    q_pt = q_fa.detach().transpose(1, 2).contiguous().requires_grad_(True)
    k_pt = k_fa.detach().transpose(1, 2).contiguous().requires_grad_(True)
    v_pt = v_fa.detach().transpose(1, 2).contiguous().requires_grad_(True)

    # Warmup Flash-Attention
    for _ in range(warmup):
        out = flash_attn_func(q_fa, k_fa, v_fa)
        out.sum().backward()
        q_fa.grad = k_fa.grad = v_fa.grad = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = flash_attn_func(q_fa, k_fa, v_fa)
        out.sum().backward()
        q_fa.grad = k_fa.grad = v_fa.grad = None
    torch.cuda.synchronize()
    fa_time = (time.perf_counter() - start) / iters * 1000

    # Warmup SDPA
    for _ in range(warmup):
        out = F.scaled_dot_product_attention(q_pt, k_pt, v_pt)
        out.sum().backward()
        q_pt.grad = k_pt.grad = v_pt.grad = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = F.scaled_dot_product_attention(q_pt, k_pt, v_pt)
        out.sum().backward()
        q_pt.grad = k_pt.grad = v_pt.grad = None
    torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / iters * 1000

    return fa_time, sdpa_time

print("\nFlash-Attention vs PyTorch SDPA")
print("=" * 60)
print(f"{'Config':<30} {'FA (ms)':<12} {'SDPA (ms)':<12} {'Speedup':<10}")
print("-" * 60)

for batch, seqlen, heads, headdim in [(2, 2048, 16, 128), (4, 4096, 16, 128)]:
    fa_ms, sdpa_ms = benchmark_comparison(batch, seqlen, heads, headdim)
    speedup = sdpa_ms / fa_ms
    config = f"B={batch}, L={seqlen}, H={heads}, D={headdim}"
    print(f"{config:<30} {fa_ms:.3f}        {sdpa_ms:.3f}        {speedup:.2f}x")
```

---

## 5. Known Issues & Troubleshooting

### Build Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `sm_103 not supported` | Old CUTLASS | `cd csrc/cutlass && git pull` |
| OOM during build | Too many jobs | `MAX_JOBS=4 NVCC_THREADS=2` |
| `nvcc not found` | CUDA_HOME not set | `export CUDA_HOME=/usr/local/cuda` |
| `compute_103f` error | CUDA < 13.0 | Falls back to `compute_103` (OK) |

### Runtime Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `No module named 'flash_attn_2_cuda'` | Not installed | Rebuild and install wheel |
| `No module named 'quack'` | Missing CuTE deps | `pip install 'quack-kernels==0.2.4'` |
| `Unsupported compute capability` | Wrong GPU | CuTE requires SM 9.x, 10.x, or 11.x |

---

## 6. Feature Support Matrix (SM103)

| Feature | Forward | Backward | Notes |
|---------|---------|----------|-------|
| BFloat16 | ✅ | ✅ | Recommended |
| Float16 | ✅ | ✅ | Supported |
| Head dim 64/96/128 | ✅ | ✅ | 128 optimal |
| Head dim 256 | ❌ | ❌ | Not supported |
| Causal | ✅ | ✅ | |
| Non-causal | ✅ | ✅ | |
| Sliding window | ✅ | ✅ | |
| Variable length | ✅ | ✅ | |
| GQA/MQA | ✅ | ✅ | |
| Block sparsity | ✅ | SM90 only | Inference only on SM103 |
| Deterministic | ✅ | ✅ | May be slower |
| FP8 | ❌ | ❌ | PR #2109 pending |

---

## 7. Future Optimizations to Test

After build verification, test these optimization opportunities:

### 7.1 Block Size Tuning (interface.py:275-287)
Add SM100/103 block size tuning - currently only SM90 has this.

### 7.2 Adaptive Seqlen Threshold
Test crossover point where CuTE becomes faster than SDPA:
- Expected: SeqLen ≥ 1024 for batch > 1
- Expected: SeqLen ≥ 1536 for batch = 1

### 7.3 GQA Backward (interface.py:711)
Check if `pack_gqa` can be enabled for backward pass.

---

## 8. Test Results Template

```
Date: _______________
GPU: NVIDIA B300 (SM 10.3)
CUDA Version: _______________
PyTorch Version: _______________
Flash-Attention Version: _______________

Build:
  [ ] SM103 gencode flags present in build.log
  [ ] 70+ SM103 kernels compiled (cuobjdump count: ___)

Functional Tests:
  [ ] Basic forward pass
  [ ] Forward + backward pass
  [ ] Variable-length sequences
  [ ] Causal attention
  [ ] CuTE interface

Performance (B=2, L=2048, H=16, D=128):
  Flash-Attention: ___ ms
  PyTorch SDPA: ___ ms
  Speedup: ___x

Notes:
_________________________________
_________________________________
```
