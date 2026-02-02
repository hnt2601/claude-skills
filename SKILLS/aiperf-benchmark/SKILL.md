---
name: aiperf-benchmark
description: "Benchmarking AI models with Nvidia Aiperf and analyzing benchmark results. Use when the user wants to run performance benchmarks on LLM inference endpoints, analyze Aiperf CSV/JSON benchmark output files, generate performance reports from benchmark data, compare model performance metrics like TTFT, ITL, throughput, or set up benchmark configurations for vLLM, TGI, or other inference servers. Triggers on keywords like aiperf, benchmark, TTFT, ITL, throughput, inference performance, model benchmark."
---

# Aiperf Benchmark Skill

Aiperf (AI Performance) is a comprehensive benchmarking tool from NVIDIA's ai-dynamo project that measures performance of generative AI models served by inference solutions.

## Installation

```bash
pip install aiperf --break-system-packages
```

## Quick Start Commands

### Basic Chat Benchmarking
```bash
aiperf profile --model <model-name> --url <server-url> --endpoint-type chat --streaming
```

### Concurrency-Based Benchmarking
```bash
aiperf profile --model <model-name> --url <server-url> --concurrency 10 --request-count 100
```

### Request Rate Benchmarking (Poisson Distribution)
```bash
aiperf profile --model <model-name> --url <server-url> --request-rate 5.0 --benchmark-duration 60
```

### Multi-Turn Conversations with ShareGPT
```bash
aiperf profile --model <model-name> --url <server-url> --public-dataset sharegpt --num-sessions 50
```

## Key CLI Options

See [references/cli_options.md](references/cli_options.md) for the complete CLI reference.

### Essential Parameters

| Parameter | Description |
|-----------|-------------|
| `-m, --model` | Model name(s) to benchmark (required) |
| `-u, --url` | Server URL (default: localhost:8000) |
| `--endpoint-type` | API type: chat, completions, embeddings, etc. |
| `--streaming` | Enable streaming responses for TTFT/ITL metrics |

### Load Configuration

| Parameter | Description |
|-----------|-------------|
| `--concurrency` | Number of concurrent requests to maintain |
| `--request-rate` | Target requests per second |
| `--request-count` | Maximum number of requests to send |
| `--benchmark-duration` | Maximum benchmark runtime in seconds |
| `--arrival-pattern` | constant, poisson (default), gamma, concurrency_burst |

### Input Configuration

| Parameter | Description |
|-----------|-------------|
| `--isl` | Mean input sequence length (tokens) |
| `--isl-stddev` | Standard deviation for input length |
| `--osl` | Mean output sequence length (tokens) |
| `--osl-stddev` | Standard deviation for output length |
| `--input-file` | Custom dataset path (JSONL) |
| `--public-dataset` | Use public dataset (e.g., sharegpt) |

### Output Configuration

| Parameter | Description |
|-----------|-------------|
| `--artifact-dir` | Output directory (default: artifacts) |
| `--export-level` | summary, records (default), or raw |
| `--slice-duration` | Duration for time-sliced analysis |

## Output Files

Aiperf generates several output files in the artifact directory:

- `profile_export_aiperf.csv` - Summary metrics in CSV
- `profile_export_aiperf.json` - Summary with metadata
- `profile_export.jsonl` - Per-request metrics
- `profile_export_raw.jsonl` - Raw request/response data (if --export-level raw)
- `*_timeslices.csv` - Time-windowed metrics (if --slice-duration set)
- `*_gpu_telemetry.jsonl` - GPU metrics (if --gpu-telemetry enabled)
- `*_server_metrics.*` - Server-side Prometheus metrics

## Analyzing Benchmark Results

Use `scripts/analyze_benchmark.py` to analyze CSV output:

```bash
python scripts/analyze_benchmark.py /path/to/profile_export_aiperf.csv
```

### Key Metrics in Output

| Metric | Description |
|--------|-------------|
| `time_to_first_token_s` | Time to first token (TTFT) |
| `inter_token_latency_s` | Inter-token latency (ITL) |
| `request_latency_s` | End-to-end request latency |
| `output_token_throughput_per_request` | Tokens/second per request |
| `input_tokens`, `output_tokens` | Token counts |
| `successful_requests`, `failed_requests` | Request status |

### CSV Analysis Workflow

1. Load the CSV with pandas
2. Filter by successful requests
3. Calculate percentiles (p50, p90, p95, p99) for latency metrics
4. Compute aggregate throughput
5. Generate comparison charts if multiple runs

```python
import pandas as pd

df = pd.read_csv('profile_export_aiperf.csv')
# Filter successful requests
df_success = df[df['request_output_error'].isna()]

# Key metrics
print(f"TTFT p50: {df_success['time_to_first_token_s'].quantile(0.5):.3f}s")
print(f"TTFT p99: {df_success['time_to_first_token_s'].quantile(0.99):.3f}s")
print(f"ITL p50: {df_success['inter_token_latency_s'].quantile(0.5)*1000:.2f}ms")
print(f"Throughput: {df_success['output_token_throughput_per_request'].mean():.1f} tok/s")
```

## Visualization

Use `aiperf plot` to generate visualizations:

```bash
aiperf plot --paths ./artifacts --output ./plots
```

Or launch interactive dashboard:

```bash
aiperf plot --dashboard --port 8050
```

## Common Benchmark Scenarios

### Latency-Focused (Interactive Use)
```bash
aiperf profile --model <model> --url <url> --streaming \
  --concurrency 1 --request-count 100 --isl 512 --osl 256
```

### Throughput-Focused (Batch Processing)
```bash
aiperf profile --model <model> --url <url> \
  --concurrency 32 --request-rate 10 --benchmark-duration 300
```

### Goodput with SLOs
```bash
aiperf profile --model <model> --url <url> --streaming \
  --concurrency 16 --goodput "request_latency:250 inter_token_latency:10"
```

### KV Cache Testing
```bash
aiperf profile --model <model> --url <url> --streaming \
  --num-prefix-prompts 10 --prefix-prompt-length 2048 \
  --isl 512 --osl 128 --concurrency 8
```

## Endpoint Types

| Type | Description |
|------|-------------|
| `chat` | OpenAI Chat Completions (default) |
| `completions` | OpenAI Completions (legacy) |
| `embeddings` | Vector embeddings generation |
| `rankings` | Passage reranking |
| `image_generation` | Image generation (FLUX.1, etc.) |
| `huggingface_generate` | HuggingFace TGI API |