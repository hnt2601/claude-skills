---
name: debug-cuda-crash
description: Debug CUDA crashes and illegal memory access errors in vLLM using core dumps, cuda-gdb, environment variables, and systematic troubleshooting
---

# Tutorial: Debugging CUDA Crashes in vLLM

This tutorial shows you how to debug CUDA crashes and errors in vLLM using CUDA core dumps, cuda-gdb, and vLLM's debugging environment variables.

## Goal

When your vLLM inference crashes with CUDA errors (illegal memory access, out-of-bounds, NaN/Inf), use these techniques to:
- Capture GPU state at the moment of crash for post-mortem analysis
- Identify the exact kernel and source line causing the error
- Debug issues hidden within CUDA graphs
- Diagnose distributed inference and NCCL communication problems

## Why CUDA Debugging is Hard in vLLM

**Problem**: CUDA errors are often asynchronously reported, producing unreliable stack traces. When using CUDA graphs (default in vLLM), the culprit kernel is obscured because the entire graph appears as a single operation.

**Solution**: CUDA core dumps capture the complete GPU state when an exception occurs, enabling precise identification of the problematic kernel even within CUDA graphs.

## Step 1: Enable vLLM Debug Logging

### Basic Debug Logging

```bash
export VLLM_LOGGING_LEVEL=DEBUG
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf
```

### Function Tracing

```bash
export VLLM_TRACE_FUNCTION=1
python my_vllm_script.py
```

This records all function calls for inspection, helping identify where crashes occur.

### Disable CUDA Graphs for Isolation

CUDA graphs can obscure the source of errors. Disable them to isolate issues:

```bash
# CLI
python -m vllm.entrypoints.openai.api_server --model my-model --enforce-eager

# Python API
from vllm import LLM
llm = LLM(model="my-model", enforce_eager=True)
```

## Step 2: Enable CUDA Core Dumps

When you encounter illegal memory access errors, enable CUDA core dumps to capture GPU state:

```bash
# Enable core dump on GPU exception
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1

# Show progress and file location
export CUDA_COREDUMP_SHOW_PROGRESS=1

# Reduce file size by skipping memory contents (still captures stack traces)
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'

# Specify where to save the coredump
export CUDA_COREDUMP_FILE="/tmp/vllm_coredump_%h.%p.%t"

# Run vLLM
python -m vllm.entrypoints.openai.api_server --model my-model
```

When a crash occurs, you'll see:
```
GPU coredump being generated...
GPU coredump generated: /tmp/vllm_coredump_hostname.12345.1
```

## Step 3: Analyze Core Dumps with cuda-gdb

Open the coredump file:

```bash
cuda-gdb -c /tmp/vllm_coredump_hostname.12345.1
```

In cuda-gdb:

```
(cuda-gdb) target cudacore /tmp/vllm_coredump_hostname.12345.1
Opening GPU coredump: /tmp/vllm_coredump_hostname.12345.1

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7f1234567890

Thread 1 received signal CUDA_EXCEPTION_14, Warp Illegal Address.
[Switching focus to CUDA kernel 0, grid 1, block (256,0,0), thread (128,0,0)]
0x00007f1234567890 in flash_fwd_kernel<...><<<(2048,1,1),(256,1,1)>>> ()
    at /path/to/csrc/flash_attn/flash_api.cu:123
```

Useful cuda-gdb commands:

```
(cuda-gdb) info cuda kernels          # List all active kernels
(cuda-gdb) info cuda threads          # Show thread coordinates
(cuda-gdb) bt                         # Backtrace with source lines
(cuda-gdb) print blockIdx             # Print block coordinates
(cuda-gdb) print threadIdx            # Print thread coordinates
```

## Step 4: Common CUDA Errors and How to Debug

### Error 1: Illegal Memory Access

**Error Message**:
```
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported...
```

**Debug Steps**:

1. **Disable CUDA graphs** to isolate the issue:
   ```bash
   python -m vllm.entrypoints.openai.api_server --model my-model --enforce-eager
   ```

2. **Enable synchronous execution** for accurate stack traces:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   python my_script.py
   ```

3. **Enable core dumps** for post-mortem analysis:
   ```bash
   export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
   export CUDA_COREDUMP_SHOW_PROGRESS=1
   python my_script.py
   ```

4. **Use compute-sanitizer** for memory checking:
   ```bash
   compute-sanitizer --tool memcheck python my_script.py
   ```

### Error 2: Out of Memory

**Error Message**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Debug Steps**:

1. **Check GPU memory usage**:
   ```bash
   nvidia-smi
   ```

2. **Reduce memory usage**:
   ```bash
   # Reduce GPU memory utilization
   python -m vllm.entrypoints.openai.api_server \
     --model my-model \
     --gpu-memory-utilization 0.8

   # Use quantization
   python -m vllm.entrypoints.openai.api_server \
     --model my-model \
     --quantization awq
   ```

3. **Use tensor parallelism** for large models:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model my-model \
     --tensor-parallel-size 2
   ```

### Error 3: NCCL Communication Failures

**Error Message**:
```
NCCL error: unhandled system error
RuntimeError: NCCL communicator was aborted
```

**Debug Steps**:

1. **Enable NCCL tracing**:
   ```bash
   export NCCL_DEBUG=TRACE
   python my_distributed_script.py
   ```

2. **Specify network interface**:
   ```bash
   export NCCL_SOCKET_IFNAME=eth0
   export GLOO_SOCKET_IFNAME=eth0
   python my_distributed_script.py
   ```

3. **Disable peer-to-peer** (for hardware issues):
   ```bash
   export NCCL_P2P_DISABLE=1
   python my_distributed_script.py
   ```

4. **Test hardware communication**:
   ```python
   # Run PyTorch distributed test
   import torch
   import torch.distributed as dist

   dist.init_process_group(backend='nccl')
   tensor = torch.ones(1024, device='cuda')
   dist.all_reduce(tensor)
   print(f"Rank {dist.get_rank()}: {tensor.sum()}")
   ```

### Error 4: CUDAGraph Errors

**Error Message**:
```
RuntimeError: CUDA error: operation not permitted when stream is capturing
```

**Debug Steps**:

1. **Disable CUDA graphs**:
   ```bash
   python -m vllm.entrypoints.openai.api_server --model my-model --enforce-eager
   ```

2. If the error disappears, the issue is related to graph capture. Enable core dumps to identify the problematic kernel within the graph.

## Step 5: Multi-Process/Multi-Node Debugging

### Set Up Host IP

```bash
export VLLM_HOST_IP=192.168.1.100
python -m vllm.entrypoints.openai.api_server --model my-model
```

### Debug Each Rank Separately

```bash
# Enable per-rank core dumps
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_FILE="/tmp/coredump_rank_%h.%p.%t"

# Enable per-rank logging
export VLLM_LOGGING_LEVEL=DEBUG

torchrun --nproc_per_node=4 my_distributed_script.py
```

This creates separate coredumps for each rank:
- `/tmp/coredump_rank_host.12345.1`
- `/tmp/coredump_rank_host.12346.2`
- etc.

## Step 6: Advanced Debugging with compute-sanitizer

### Memory Checker

```bash
compute-sanitizer --tool memcheck python my_script.py
```

Output:
```
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x1234 in flash_fwd_kernel<float>
=========     by thread (256,0,0) in block (10,0,0)
=========     Address 0x7f1234567890 is out of bounds
```

### Race Condition Checker

```bash
compute-sanitizer --tool racecheck python my_script.py
```

### Synchronization Checker

```bash
compute-sanitizer --tool synccheck python my_script.py
```

## Step 7: Enable Debug Symbols for Source-Level Debugging

Compile vLLM with debug symbols for precise line-level error attribution:

```bash
export NVCC_PREPEND_FLAGS='-lineinfo'
pip install vllm --no-build-isolation
```

Now cuda-gdb will show exact source lines:

```
(cuda-gdb) bt
#0  flash_fwd_kernel<...> at flash_attn/flash_api.cu:123
#1  at_cuda_detail::launch_kernel at ATen/cuda/CUDAContext.cpp:456
```

## Step 8: Kernel-Level Debugging with printf()

For custom CUDA kernels, use `printf()` for debugging:

```cpp
__global__ void MyKernel(const float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Print from one thread to avoid spam
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("n=%d, input[0]=%f\n", n, input[0]);
  }

  if (idx < n) {
    output[idx] = input[idx] * 2.0f;
  }
}
```

**Important**: Flush printf buffer after kernel:
```python
my_kernel(input, output)
torch.cuda.synchronize()  # Flushes printf output
```

## Environment Variables Reference

### vLLM Debugging Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `VLLM_LOGGING_LEVEL` | `DEBUG` | Enable detailed logging |
| `VLLM_TRACE_FUNCTION` | `1` | Record all function calls |
| `VLLM_HOST_IP` | IP address | Override auto-detected IP |

### CUDA Debugging Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `CUDA_LAUNCH_BLOCKING` | `1` | Synchronous kernel execution |
| `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` | `1` | Enable GPU coredumps |
| `CUDA_COREDUMP_SHOW_PROGRESS` | `1` | Show coredump generation progress |
| `CUDA_COREDUMP_FILE` | path | Coredump file location (%h=host, %p=pid, %t=time) |
| `CUDA_COREDUMP_GENERATION_FLAGS` | flags | Control what to include in coredump |
| `CUDA_DEVICE_WAITS_ON_EXCEPTION` | `1` | Halt GPU for live debugger attachment |

### NCCL Debugging Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `NCCL_DEBUG` | `TRACE` | Enable detailed NCCL logging |
| `NCCL_SOCKET_IFNAME` | interface | Specify network interface |
| `NCCL_P2P_DISABLE` | `1` | Disable peer-to-peer communication |
| `NCCL_CUMEM_ENABLE` | `0` | Disable cuMem allocator (vLLM default) |

## Best Practices

### 1. Start with CUDA_LAUNCH_BLOCKING

```bash
export CUDA_LAUNCH_BLOCKING=1
```

This makes CUDA errors synchronous, providing more accurate stack traces.

### 2. Disable CUDA Graphs for Debugging

```bash
python -m vllm.entrypoints.openai.api_server --model my-model --enforce-eager
```

CUDA graphs obscure the source of errors. Disable them first.

### 3. Enable Core Dumps for Hard Crashes

```bash
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
```

Core dumps provide complete GPU state for post-mortem analysis.

### 4. Use Memory-Reduced Coredumps

```bash
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
```

This keeps coredumps small while preserving stack traces.

### 5. Avoid Production Use

Debug settings have significant performance overhead. Remove all debug environment variables in production.

## Troubleshooting

### No Coredump Generated

**Problem**: `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` set but no coredump appears

**Solutions**:
1. Check write permissions on coredump directory
2. Verify sufficient disk space
3. Some exception types don't trigger coredumps (use compute-sanitizer instead)

### Coredump Too Large

**Problem**: Coredump file is multiple gigabytes

**Solution**: Use generation flags to skip memory contents:
```bash
export CUDA_COREDUMP_GENERATION_FLAGS='skip_global_memory,skip_shared_memory,skip_local_memory'
```

### cuda-gdb Missing Source Lines

**Problem**: Stack trace shows addresses instead of source lines

**Solution**: Recompile with debug symbols:
```bash
export NVCC_PREPEND_FLAGS='-lineinfo'
pip install vllm --no-build-isolation
```

### Model Download Hangs

**Problem**: vLLM hangs when downloading model

**Solution**: Pre-download the model:
```bash
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-2-7b
python -m vllm.entrypoints.openai.api_server --model ./models/llama-2-7b
```

## Quick Examples

### Debug Illegal Memory Access

```bash
# Step 1: Enable synchronous execution
export CUDA_LAUNCH_BLOCKING=1

# Step 2: Disable CUDA graphs
python -m vllm.entrypoints.openai.api_server --model my-model --enforce-eager

# If crash persists, enable core dumps:
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
python -m vllm.entrypoints.openai.api_server --model my-model --enforce-eager

# Analyze coredump:
cuda-gdb -c /tmp/coredump_*
```

### Debug NCCL Issues

```bash
export NCCL_DEBUG=TRACE
export NCCL_SOCKET_IFNAME=eth0
export VLLM_LOGGING_LEVEL=DEBUG
python -m vllm.entrypoints.openai.api_server --model my-model --tensor-parallel-size 4
```

### Debug with compute-sanitizer

```bash
compute-sanitizer --tool memcheck python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='gpt2', enforce_eager=True)
output = llm.generate(['Hello'], SamplingParams(max_tokens=10))
print(output)
"
```

## Example: Full Debug Session

### Your vLLM server crashes:

```
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call
```

### Step 1: Enable synchronous execution

```bash
export CUDA_LAUNCH_BLOCKING=1
python -m vllm.entrypoints.openai.api_server --model my-model
```

### Step 2: Disable CUDA graphs

```bash
python -m vllm.entrypoints.openai.api_server --model my-model --enforce-eager
```

### Step 3: Enable core dumps

```bash
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_FILE="/tmp/vllm_crash_%p"
python -m vllm.entrypoints.openai.api_server --model my-model --enforce-eager
```

### Step 4: Analyze the coredump

```bash
cuda-gdb -c /tmp/vllm_crash_12345
(cuda-gdb) target cudacore /tmp/vllm_crash_12345
(cuda-gdb) info cuda kernels
(cuda-gdb) bt
```

### Step 5: Found the issue

```
#0  flash_fwd_kernel<...> at flash_attn/flash_api.cu:456
    with invalid tensor shape: expected (batch, heads, seq, dim) but got (batch, seq, heads, dim)
```

### Step 6: Fix and verify

Fix the tensor shape issue in your code and verify the fix runs without errors.

## Summary

1. **Start with basic debugging**:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   export VLLM_LOGGING_LEVEL=DEBUG
   ```

2. **Disable CUDA graphs** to isolate issues:
   ```bash
   --enforce-eager
   ```

3. **Enable core dumps** for hard crashes:
   ```bash
   export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
   ```

4. **Analyze with cuda-gdb**:
   ```bash
   cuda-gdb -c /path/to/coredump
   ```

5. **Clean up** in production:
   ```bash
   unset CUDA_LAUNCH_BLOCKING
   unset CUDA_ENABLE_COREDUMP_ON_EXCEPTION
   ```

## Related Documentation

- [vLLM Debugging Guide](https://docs.vllm.ai/en/latest/getting_started/debugging.html)
- [CUDA Core Dumps Blog Post](https://blog.vllm.ai/2025/08/11/cuda-debugging.html)
- [NVIDIA cuda-gdb Documentation](https://docs.nvidia.com/cuda/cuda-gdb/)
- [NVIDIA Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/)
