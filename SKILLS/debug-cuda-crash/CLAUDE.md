# vLLM CUDA Debugging Skill

This skill provides guidance for debugging CUDA crashes and errors in vLLM inference deployments.

## Quick Reference

### Essential Environment Variables

```bash
# Basic debugging
export CUDA_LAUNCH_BLOCKING=1
export VLLM_LOGGING_LEVEL=DEBUG

# CUDA core dumps for crash analysis
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_FILE="/tmp/vllm_crash_%h.%p.%t"

# NCCL debugging for distributed inference
export NCCL_DEBUG=TRACE
```

### Debugging Workflow

1. **Disable CUDA graphs**: `--enforce-eager`
2. **Enable synchronous execution**: `CUDA_LAUNCH_BLOCKING=1`
3. **Enable core dumps**: `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`
4. **Analyze with cuda-gdb**: `cuda-gdb -c /path/to/coredump`

## When to Use This Skill

- Debugging "illegal memory access" errors in vLLM
- Analyzing crashes within CUDA graphs
- Troubleshooting NCCL communication failures in multi-GPU setups
- Investigating out-of-memory issues
- Post-mortem analysis of GPU crashes

## Key Tools

- **cuda-gdb**: GPU debugger for analyzing core dumps
- **compute-sanitizer**: Memory checker, race condition detector
- **nvidia-smi**: GPU monitoring and memory usage

## References

- [SKILL.md](SKILL.md) - Complete debugging tutorial
- [vLLM Debugging Docs](https://docs.vllm.ai/en/latest/getting_started/debugging.html)
- [CUDA Core Dumps Blog](https://blog.vllm.ai/2025/08/11/cuda-debugging.html)
