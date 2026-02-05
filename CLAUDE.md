# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashAttention is a memory-efficient attention algorithm that reduces memory from O(N²) to O(N) through tiling and recomputation. This repository contains multiple implementation paths:

- **FA2 (v2.x)**: C++/CUDA with CUTLASS, stable and broadly compatible (SM 8.0+)
- **FA3 (v3.x)**: Hopper-optimized in `hopper/` directory (H100/H800, requires CUDA ≥12.3)
- **CuTE**: Python DSL with JIT compilation, latest optimizations for SM 9.0+ (Hopper/Blackwell)
- **ROCm**: AMD support via Composable Kernel (CK) and Triton backends

**B300 Optimization Target**: This codebase is optimized for B300 GPUs (SM 10.3, Blackwell). The build system includes explicit SM103 gencode support. SM103 uses SM100 kernel code paths in CuTE—the `10.x` family shares implementations.

## Build Commands

```bash
# Standard installation (attempts prebuilt wheels first)
pip install flash-attn --no-build-isolation

# Force build from source
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation

# Memory-constrained systems
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Editable install for development
pip install -e . --no-build-isolation --no-deps

# B300-specific build
export FLASH_ATTN_CUDA_ARCHS="103"  # B300-only
export MAX_JOBS=8
export NVCC_THREADS=4
pip wheel . --no-build-isolation --no-deps -v

# FlashAttention-3 (Hopper)
cd hopper && python setup.py install
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `FLASH_ATTN_CUDA_ARCHS` | Target GPU architectures | `"80;90;103"` |
| `FLASH_ATTENTION_FORCE_BUILD` | Force local compilation | `TRUE` |
| `FLASH_ATTENTION_SKIP_CUDA_BUILD` | Skip CUDA kernel compilation | `TRUE` |
| `MAX_JOBS` | Limit parallel compilation jobs | `4` |
| `NVCC_THREADS` | Control NVCC threading | `4` |

## Running Tests

```bash
# FlashAttention v2.x tests
pytest -q -s tests/test_flash_attn.py

# Single test pattern
pytest -q -s tests/test_flash_attn.py -k "causal"
pytest -q -s tests/test_flash_attn.py::test_flash_attn_output -k "fp16_seq128"

# FlashAttention v3.x (Hopper)
cd hopper
export PYTHONPATH=$PWD
pytest -q -s test_flash_attn.py

# ROCm CK backend
pytest tests/test_flash_attn_ck.py

# ROCm Triton backend
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pytest tests/test_flash_attn_triton_amd.py

# With autotuning (better performance, slower)
FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE" pytest tests/test_flash_attn_triton_amd.py
```

## Code Architecture

### Directory Structure

```
flash-attention/
├── flash_attn/                  # Main v2.x library
│   ├── __init__.py              # Public API exports
│   ├── flash_attn_interface.py  # Core Python interface
│   ├── cute/                    # CuTE DSL implementation (SM 9.0+)
│   │   ├── interface.py         # Main API, autograd integration
│   │   ├── flash_fwd_sm100.py   # Blackwell forward kernel
│   │   ├── flash_bwd_sm100.py   # Blackwell backward kernel
│   │   └── blackwell_helpers.py # tcgen05 MMA helpers
│   ├── modules/                 # High-level layers (MHA, MLP, etc.)
│   └── models/                  # Full model implementations (GPT, BERT, LLaMA)
├── hopper/                      # FlashAttention v3.x (Hopper-optimized)
│   ├── flash_attn_interface.py  # v3.x public API
│   ├── setup.py                 # Advanced build with feature flags
│   └── instantiations/          # 450+ kernel variant files
├── csrc/                        # C++ and CUDA source
│   ├── flash_attn/              # v2.x CUDA kernels
│   └── flash_attn_ck/           # ROCm Composable Kernel
└── tests/                       # Test suite
```

### Public API (flash_attn)

```python
from flash_attn import (
    flash_attn_func,              # Core: Q, K, V separate tensors
    flash_attn_qkvpacked_func,    # QKV stacked in single tensor
    flash_attn_kvpacked_func,     # KV stacked together
    flash_attn_varlen_func,       # Variable-length sequences
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_with_kvcache,      # Incremental decoding with KV cache
)
```

### Tensor Layout Convention

**Standard flash_attn**: `[batch, seqlen, heads, headdim]` = `[B, L, H, D]`

**CuTE interface** expects the same layout. If converting from `[B, H, L, D]`:
```python
q_cute = q.transpose(1, 2).contiguous()  # CRITICAL: .contiguous() is mandatory
```

### CUDA Kernel Naming Pattern

Files in `csrc/flash_attn/src/`:
```
flash_{fwd|bwd}_hdim{32|64|96|128|192|256}_{fp16|bf16}[_causal][_split][_softcap]_sm{80|90}.cu
```

### CuTE Architecture Selection

In `flash_attn/cute/interface.py`:
```python
compute_capability = torch.cuda.get_device_capability()[0]
# 9  -> SM90 (Hopper)
# 10 -> SM100/SM103 (Blackwell) <- B300 uses this
# 11 -> SM110 (Thor)
```

## B300/SM103 Specifics

### Feature Support Matrix (SM100/SM103)

| Feature | Forward | Backward | Notes |
|---------|---------|----------|-------|
| BFloat16/Float16 | ✅ | ✅ | Recommended: bf16 for training |
| Head dim 64/96/128 | ✅ | ✅ | 128 is Flux.2-dev default |
| Head dim 256 | ❌ | ❌ | Not yet supported |
| Variable length (varlen) | ✅ | ✅ | Now works on SM100/SM103 |
| Causal/Non-causal | ✅ | ✅ | |
| Sliding window | ✅ | ✅ | SM100+ only for bwd |
| GQA/MQA | ✅ | ✅ | With pack_gqa optimization |
| Block sparsity bwd | ❌ | SM90 only | Use non-sparse for training |
| FP8 | ❌ | ❌ | PR #2109 open |

### Blackwell tcgen05 MMA

B300 uses `tcgen05.mma` instructions:
- 256×256×16 MMA operations spanning 2 SMs
- Operands in shared memory and Tensor Memory (TMEM)
- Single-thread semantics (simpler than Hopper's WGMMA)

### SM103 Build Configuration

In `setup.py`, ensure SM103 support:
```python
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;103;110;120").split(";")
```

### CuTE Dependencies for Blackwell

```bash
pip install 'quack-kernels==0.2.4'  # Installs CUTLASS DSL, TVM FFI, DLPack
```

### Performance Crossover

| Sequence Length | Recommendation |
|-----------------|----------------|
| < 1024 | Use SDPA (CuTE JIT overhead dominates) |
| 1024-2048 | CuTE or FA2 both good |
| > 2048 | CuTE (Blackwell optimizations shine) |

## Code Style

- Line length: 100 characters
- Python: 3.9+ (core), 3.8+ (tests)
- Formatter: Ruff (replaces Black)
- Pre-commit: Ruff linting for `flash_attn/cute/` only

## Key Dependencies

- PyTorch ≥2.2 (≥2.9 recommended for Blackwell)
- CUDA ≥12.0 (≥12.8 for FA3, ≥13.0 for SM103 compute_103f)
- `packaging`, `psutil`, `ninja` (build)
- `einops` (runtime)
- CUTLASS DSL ≥4.3.4 for CuTE (4.3.1+ for SM103 support)
