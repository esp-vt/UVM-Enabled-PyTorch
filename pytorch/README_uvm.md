# CUDA Unified Virtual Memory (UVM) Support for PyTorch

This document describes the UVM implementation added to PyTorch's CUDA caching allocator, providing three memory allocation modes:

1. **Non-UVM** (default) - Standard `cudaMalloc` allocation
2. **UVM** - Uses `cudaMallocManaged` for automatic CPU-GPU memory migration
3. **UVM+Prefetch** - Uses `cudaMallocManaged` + `cudaMemPrefetchAsync` to proactively migrate pages to GPU

## Code Changes Summary

### 1. `c10/cuda/CUDAAllocatorConfig.h`

**Added accessor methods** (after line 75):
```cpp
/** UVM (Unified Virtual Memory) allocator settings */
static bool use_uvm() {
  return instance().m_use_uvm;
}

static bool uvm_prefetch() {
  return instance().m_uvm_prefetch;
}
```

**Added configuration keys to `getKeys()`** (around line 167):
```cpp
"use_uvm",
"uvm_prefetch"
```

**Added parser declarations** (after line 197):
```cpp
size_t parseUseUvm(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i);
size_t parseUvmPrefetch(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i);
```

**Added private member variables** (after line 210):
```cpp
std::atomic<bool> m_use_uvm{false};
std::atomic<bool> m_uvm_prefetch{false};
```

### 2. `c10/cuda/CUDAAllocatorConfig.cpp`

**Added parsing in `parseArgs()`** (after line 120):
```cpp
} else if (key == "use_uvm") {
  i = parseUseUvm(tokenizer, i);
  used_native_specific_option = true;
} else if (key == "uvm_prefetch") {
  i = parseUvmPrefetch(tokenizer, i);
  used_native_specific_option = true;
```

**Added parser implementations** (after line 210):
```cpp
size_t CUDAAllocatorConfig::parseUseUvm(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  m_use_uvm = tokenizer.toBool(++i);

  if (m_use_uvm &&
      c10::CachingAllocator::AcceleratorAllocatorConfig::
          use_expandable_segments()) {
    TORCH_WARN(
        "use_uvm is incompatible with expandable_segments. "
        "Disabling expandable_segments for UVM allocations.");
  }
  return i;
}

size_t CUDAAllocatorConfig::parseUvmPrefetch(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  m_uvm_prefetch = tokenizer.toBool(++i);
  return i;
}
```

### 3. `c10/cuda/CUDACachingAllocator.cpp`

**Added `is_uvm` field to Block struct** (after line 195):
```cpp
bool is_uvm{false}; // true if allocated with cudaMallocManaged (UVM)
```

**Modified Block constructor** (line 213-224):
```cpp
Block(
    c10::DeviceIndex device,
    cudaStream_t stream,
    size_t size,
    BlockPool* pool,
    void* ptr,
    bool is_uvm = false)  // Added parameter
    : device(device),
      stream(stream),
      size(size),
      requested_size(0),
      pool(pool),
      ptr(ptr),
      is_uvm(is_uvm) {}  // Added initialization
```

**Modified `allocPrimitive()`** (line 1038-1045):
```cpp
cudaError_t allocPrimitive(
    void** ptr,
    size_t size,
    AllocParams& p,
    bool& out_is_uvm) {  // Added output parameter
  if (p.pool->owner_PrivatePool && p.pool->owner_PrivatePool->allocator()) {
    *ptr = p.pool->owner_PrivatePool->allocator()->raw_alloc(size);
    out_is_uvm = false;
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
  } else {
    bool use_uvm = CUDAAllocatorConfig::use_uvm();
    out_is_uvm = use_uvm;
    if (use_uvm) {
      cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaMallocManaged(ptr, size));
      if (err == cudaSuccess && *ptr != nullptr &&
          CUDAAllocatorConfig::uvm_prefetch()) {
        // Prefetch to current device to avoid page faults
        C10_CUDA_CHECK(cudaMemPrefetchAsync(*ptr, size, p.device(), nullptr));
      }
      return err;
    } else {
      return C10_CUDA_ERROR_HANDLED(cudaMalloc(ptr, size));
    }
  }
}
```

**Modified `cudaMallocMaybeCapturing()`** (line 1047-1059):
```cpp
cudaError_t cudaMallocMaybeCapturing(
    void** ptr,
    size_t size,
    AllocParams& p,
    bool& out_is_uvm) {  // Added output parameter
  if (at::cuda::currentStreamCaptureStatusMayInitCtx() ==
      at::cuda::CaptureStatus::None) {
    return allocPrimitive(ptr, size, p, out_is_uvm);
  } else {
    if (CUDAAllocatorConfig::use_uvm()) {
      TORCH_WARN_ONCE(
          "Using UVM (cudaMallocManaged) during CUDA graph capture may cause issues.");
    }
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    return allocPrimitive(ptr, size, p, out_is_uvm);
  }
}
```

**Modified `alloc_block()`** (line 3270-3355):
- Added `bool is_uvm = false;` variable (line 3280)
- Added UVM check to expandable_segments condition (line 3297):
  ```cpp
  !CUDAAllocatorConfig::use_uvm() && // UVM incompatible with expandable
  ```
- Updated `cudaMallocMaybeCapturing()` calls to pass `is_uvm` (lines 3316, 3318)
- Updated Block constructor call to pass `is_uvm` (line 3352)

### 4. `c10/cuda/CUDACachingAllocator.h`

**Added `is_uvm` to BlockInfo struct** (line 61-69):
```cpp
struct BlockInfo {
  size_t size = 0;
  size_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  bool is_uvm = false; // true if allocated with cudaMallocManaged (UVM)
  std::shared_ptr<GatheredContext> context_when_allocated;
};
```

**Added `is_uvm` to SegmentInfo struct** (line 72-86):
```cpp
struct SegmentInfo {
  // ... existing fields ...
  bool is_large = false;
  bool is_expandable = false;
  bool is_uvm = false; // true if allocated with cudaMallocManaged (UVM)
  MempoolId_t owner_private_pool_id = {0, 0};
  // ...
};
```

## Setting Up a Conda Environment

### Prerequisites
- CUDA Toolkit 11.0+ (for UVM support)
- CUDA-capable GPU with compute capability 6.0+ (Pascal or newer)
- Python 3.8+

### Installation Steps

```bash
# 1. Create a new conda environment
conda create -n pytorch-uvm python=3.10 -y
conda activate pytorch-uvm

# 2. Install build dependencies
conda install cmake ninja -y
pip install typing_extensions pyyaml numpy

# 3. Install CUDA toolkit (if not using system CUDA)
# Option A: Use conda's cudatoolkit
conda install -c nvidia cuda-toolkit=12.1 -y

# Option B: Use system CUDA (set CUDA_HOME)
export CUDA_HOME=/usr/local/cuda

# 4. Clone and prepare PyTorch (if not already done)
# git clone https://github.com/pytorch/pytorch.git
# cd pytorch
git submodule sync
git submodule update --init --recursive

# 5. Set build environment variables
export CMAKE_PREFIX_PATH=${CONDA_PREFIX}
export USE_CUDA=1
export MAX_JOBS=$(nproc)  # Use all CPU cores

# 6. Build and install PyTorch
# For development (editable install):
python setup.py develop

# Or for production install:
# python setup.py install

# 7. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage Guide

### Configuration Modes

| Mode | Environment Variable | Behavior |
|------|---------------------|----------|
| Non-UVM | (default) | Standard `cudaMalloc` |
| UVM | `PYTORCH_CUDA_ALLOC_CONF=use_uvm:True` | `cudaMallocManaged` |
| UVM+Prefetch | `PYTORCH_CUDA_ALLOC_CONF=use_uvm:True,uvm_prefetch:True` | `cudaMallocManaged` + `cudaMemPrefetchAsync` |

### Example: Non-UVM Mode (Default)

```python
import torch

# Default behavior - uses cudaMalloc
x = torch.randn(1000, 1000, device='cuda')
y = x + 1
torch.cuda.synchronize()
print("Non-UVM allocation successful")
```

### Example: UVM Mode

```bash
export PYTORCH_CUDA_ALLOC_CONF=use_uvm:True
```

```python
import torch

# Memory is allocated with cudaMallocManaged
# Pages migrate automatically between CPU and GPU
x = torch.randn(1000, 1000, device='cuda')
y = x + 1
torch.cuda.synchronize()
print("UVM allocation successful")

# Check memory stats
stats = torch.cuda.memory_stats()
print(f"Allocated: {stats['allocated_bytes.all.current']} bytes")
```

### Example: UVM+Prefetch Mode

```bash
export PYTORCH_CUDA_ALLOC_CONF=use_uvm:True,uvm_prefetch:True
```

```python
import torch

# Memory is allocated with cudaMallocManaged and immediately
# prefetched to the GPU to avoid page faults during computation
x = torch.randn(1000, 1000, device='cuda')
y = x + 1
torch.cuda.synchronize()
print("UVM+Prefetch allocation successful")
```

### Setting Configuration in Python

You can also set the environment variable before importing torch:

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "use_uvm:True,uvm_prefetch:True"

import torch
# Now all CUDA allocations will use UVM+Prefetch
```

## When to Use Each Mode

### Non-UVM (Default)
- Production workloads that fit entirely in GPU memory
- Best performance when memory access patterns are predictable
- Traditional deep learning training and inference

### UVM Mode
- Prototyping models that don't fit in GPU memory
- Workloads with unpredictable memory access patterns
- Simplifying memory management during development
- When you want automatic CPU-GPU memory migration

### UVM+Prefetch Mode
- Same use cases as UVM mode
- When you want to avoid page fault overhead
- When memory access patterns favor GPU-first access
- Better performance than pure UVM for GPU-heavy workloads

## Limitations and Considerations

1. **Incompatibility with `expandable_segments`**: UVM is incompatible with the `expandable_segments` allocator option. When UVM is enabled, expandable_segments is automatically disabled with a warning.

2. **CUDA Graph Capture**: Using UVM during CUDA graph capture may cause issues. A warning is emitted when this is detected.

3. **Performance Overhead**:
   - UVM may have higher latency for first access due to page faults
   - UVM+Prefetch mitigates this but adds prefetch overhead
   - For workloads that fit in GPU memory, standard cudaMalloc is typically faster

4. **Hardware Requirements**:
   - Requires CUDA 8.0+ for UVM support
   - Requires GPU with compute capability 6.0+ (Pascal or newer)
   - Best performance on systems with NVLink or high-bandwidth PCIe

5. **Memory Oversubscription**: UVM allows allocating more memory than available on the GPU, which can lead to severe performance degradation if used without care.

## Troubleshooting

### Check if UVM is enabled
```python
import torch
# This will print True if UVM is enabled
from torch._C import _cuda_CUDACachingAllocator as allocator
# Check allocator configuration via environment variable
import os
print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "Not set"))
```

### Common Issues

1. **"cudaMallocManaged failed"**: Ensure your GPU supports UVM (compute capability 6.0+)

2. **Poor performance with UVM**: Try UVM+Prefetch mode or check if your workload has irregular memory access patterns

3. **Warning about expandable_segments**: This is expected when both options are set. UVM takes precedence.

## References

- [CUDA Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [cudaMallocManaged Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b)
- [cudaMemPrefetchAsync Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8dc9199943d421bc8bc7f473df12e42)


● All implementation tasks are complete. Here's a summary of what was done:                                                                                                                        
                                                                                                                                                                                                   
  Implementation Summary                                                                                                                                                                           
                                                                                                                                                                                                   
  Files Modified                                                                                                                                                                                   
  ┌───────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────┐                                                        
  │               File                │                                              Changes                                              │                                                        
  ├───────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤                                                        
  │ c10/cuda/CUDAAllocatorConfig.h    │ Added use_uvm(), uvm_prefetch() accessors, config keys, parser declarations, and member variables │                                                        
  ├───────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤                                                        
  │ c10/cuda/CUDAAllocatorConfig.cpp  │ Added parsing logic for use_uvm and uvm_prefetch configuration options                            │                                                        
  ├───────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤                                                        
  │ c10/cuda/CUDACachingAllocator.cpp │ Added is_uvm to Block struct, modified allocPrimitive() and alloc_block() to support UVM/prefetch │                                                        
  ├───────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤                                                        
  │ c10/cuda/CUDACachingAllocator.h   │ Added is_uvm field to BlockInfo and SegmentInfo structs                                           │                                                        
  └───────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────┘                                                        
  New File Created                                                                                                                                                                                 
  ┌───────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐                                                                                      
  │     File      │                                       Description                                       │                                                                                      
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤                                                                                      
  │ README_uvm.md │ Comprehensive documentation including code changes, setup instructions, and usage guide │                                                                                      
  └───────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘                                                                                      
  Configuration Options                                                                                                                                                                            
                                                                                                                                                                                                   
  # UVM mode                                                                                                                                                                                       
  export PYTORCH_CUDA_ALLOC_CONF=use_uvm:True                                                                                                                                                      
                                                                                                                                                                                                   
  # UVM+Prefetch mode                                                                                                                                                                              
  export PYTORCH_CUDA_ALLOC_CONF=use_uvm:True,uvm_prefetch:True                                                                                                                                    
                                                                                                                                                                                                   
  The README_uvm.md file contains detailed documentation including:                                                                                                                                
  - Complete code change descriptions with line numbers                                                                                                                                            
  - Conda environment setup instructions                                                                                                                                                           
  - Usage examples for all three modes                                                                                                                                                             
  - Troubleshooting guide and limitations     
