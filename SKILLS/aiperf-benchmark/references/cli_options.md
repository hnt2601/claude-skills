# Aiperf CLI Options Reference

## Commands

- `aiperf analyze-trace` - Analyze mooncake trace for prefix statistics
- `aiperf profile` - Run benchmarks on inference endpoints
- `aiperf plot` - Generate visualizations from profiling data

## aiperf profile

### Endpoint Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Model name(s) to benchmark (required) | - |
| `--model-selection-strategy` | round_robin or random | round_robin |
| `--endpoint, --custom-endpoint` | Custom API path (e.g., `/v1/custom`) | - |
| `--endpoint-type` | chat, completions, embeddings, rankings, etc. | chat |
| `--streaming` | Enable streaming responses | false |
| `-u, --url` | Server URL | localhost:8000 |
| `--request-timeout-seconds` | Max request timeout | 21600 |
| `--api-key` | API authentication key | - |
| `--transport` | Transport protocol (http) | auto |
| `--use-legacy-max-tokens` | Use 'max_tokens' instead of 'max_completion_tokens' | false |
| `--use-server-token-count` | Use server token counts instead of client tokenization | false |
| `--connection-reuse-strategy` | pooled, never, sticky-user-sessions | pooled |

### Input Options

| Option | Description | Default |
|--------|-------------|---------|
| `--extra-inputs` | Additional API request parameters | - |
| `-H, --header` | Custom HTTP headers | - |
| `--input-file` | Dataset file path (JSONL) | - |
| `--fixed-schedule` | Replay timestamps from dataset | false |
| `--public-dataset` | Public dataset (sharegpt) | - |
| `--custom-dataset-type` | single_turn, multi_turn, random_pool, mooncake_trace | - |
| `--dataset-sampling-strategy` | sequential, random, shuffle | - |
| `--random-seed` | Seed for reproducibility | - |
| `--goodput` | SLO specifications (e.g., "request_latency:250") | - |

### Input Sequence Length (ISL)

| Option | Description | Default |
|--------|-------------|---------|
| `--isl, --prompt-input-tokens-mean` | Mean input tokens | 550 |
| `--isl-stddev, --prompt-input-tokens-stddev` | Input token std dev | 0 |
| `--isl-block-size` | Token block size for KV cache | 512 |
| `--seq-dist` | ISL,OSL distribution pairs | - |

### Output Sequence Length (OSL)

| Option | Description | Default |
|--------|-------------|---------|
| `--osl, --prompt-output-tokens-mean` | Mean output tokens | - |
| `--osl-stddev, --prompt-output-tokens-stddev` | Output token std dev | 0 |

### Prefix Prompt (KV Cache Testing)

| Option | Description | Default |
|--------|-------------|---------|
| `--num-prefix-prompts` | Number of prefix prompts | 0 |
| `--prefix-prompt-length` | Tokens per prefix | 0 |
| `--shared-system-prompt-length` | Shared system prompt length | - |
| `--user-context-prompt-length` | Per-session context length | - |

### Multimodal Input

#### Audio
| Option | Description | Default |
|--------|-------------|---------|
| `--audio-batch-size` | Audio files per request | 1 |
| `--audio-length-mean` | Mean audio duration (seconds) | 0 |
| `--audio-length-stddev` | Audio duration std dev | 0 |
| `--audio-format` | wav, mp3 | wav |

#### Image
| Option | Description | Default |
|--------|-------------|---------|
| `--image-batch-size` | Images per request | 1 |
| `--image-width-mean` | Mean image width (pixels) | 0 |
| `--image-height-mean` | Mean image height (pixels) | 0 |
| `--image-format` | png, jpeg, random | png |

#### Video
| Option | Description | Default |
|--------|-------------|---------|
| `--video-batch-size` | Videos per request | 1 |
| `--video-duration` | Video duration (seconds) | 5.0 |
| `--video-fps` | Frames per second | 4 |
| `--video-format` | mp4, webm | webm |
| `--video-codec` | FFmpeg codec | libvpx-vp9 |

### Load Generator

| Option | Description | Default |
|--------|-------------|---------|
| `--benchmark-duration` | Max runtime (seconds) | - |
| `--benchmark-grace-period` | Grace period after duration | 30 |
| `--concurrency` | Concurrent requests | - |
| `--prefill-concurrency` | Max concurrent prefill requests | - |
| `--request-rate` | Requests per second | - |
| `--arrival-pattern` | constant, poisson, gamma, concurrency_burst | poisson |
| `--arrival-smoothness` | Gamma distribution shape | 1.0 |
| `--request-count` | Max requests | - |

### Warmup Options

| Option | Description |
|--------|-------------|
| `--warmup-request-count` | Warmup requests |
| `--warmup-duration` | Warmup duration (seconds) |
| `--warmup-concurrency` | Warmup concurrency |
| `--warmup-request-rate` | Warmup request rate |
| `--warmup-grace-period` | Warmup grace period |

### Ramp Options

| Option | Description |
|--------|-------------|
| `--concurrency-ramp-duration` | Concurrency ramp time (seconds) |
| `--request-rate-ramp-duration` | Request rate ramp time |

### Conversation/Session Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num-sessions` | Total conversations | - |
| `--num-dataset-entries` | Unique dataset entries | 100 |
| `--session-turns-mean` | Mean turns per conversation | 1 |
| `--session-turns-stddev` | Turn count std dev | 0 |
| `--session-turn-delay-mean` | Mean delay between turns (ms) | 0 |
| `--session-turn-delay-stddev` | Turn delay std dev | 0 |

### Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--artifact-dir` | Output directory | artifacts |
| `--profile-export-file` | Export file prefix | - |
| `--export-level` | summary, records, raw | records |
| `--slice-duration` | Time-slice duration (seconds) | - |
| `--export-http-trace` | Include HTTP trace data | false |
| `--show-trace-timing` | Show HTTP timing in console | false |

### Tokenizer Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tokenizer` | HuggingFace tokenizer ID | model name |
| `--tokenizer-revision` | Tokenizer version | main |
| `--tokenizer-trust-remote-code` | Allow custom tokenizer code | false |

### Telemetry Options

| Option | Description |
|--------|-------------|
| `--gpu-telemetry` | Enable GPU telemetry |
| `--no-gpu-telemetry` | Disable GPU telemetry |
| `--server-metrics` | Prometheus endpoint URLs |
| `--no-server-metrics` | Disable server metrics |
| `--server-metrics-formats` | json, csv, jsonl, parquet |

### Service Options

| Option | Description | Default |
|--------|-------------|---------|
| `--log-level` | TRACE, DEBUG, INFO, WARNING, ERROR | INFO |
| `-v, --verbose` | Enable DEBUG logging | - |
| `-vv, --extra-verbose` | Enable TRACE logging | - |
| `--ui-type` | none, simple, dashboard | dashboard |
| `--workers-max` | Max worker processes | - |

## aiperf analyze-trace

| Option | Description | Default |
|--------|-------------|---------|
| `--input-file` | Mooncake trace JSONL (required) | - |
| `--block-size` | KV cache block size | 512 |
| `--output-file` | Analysis report output (JSON) | - |

## aiperf plot

| Option | Description | Default |
|--------|-------------|---------|
| `--paths` | Profiling run directories | ./artifacts |
| `--output` | Output directory for plots | /plots |
| `--theme` | light or dark | light |
| `--config` | Custom config YAML | ~/.aiperf/plot_config.yaml |
| `--verbose` | Show error tracebacks | false |
| `--dashboard` | Launch interactive server | false |
| `--host` | Dashboard host | 127.0.0.1 |
| `--port` | Dashboard port | 8050 |

## Endpoint Types

| Type | Description |
|------|-------------|
| `chat` | OpenAI Chat Completions API (multi-modal, streaming) |
| `completions` | OpenAI Completions API (legacy) |
| `embeddings` | OpenAI Embeddings API |
| `rankings` | Passage reranking APIs |
| `cohere_rankings` | Cohere Rerank API |
| `hf_tei_rankings` | HuggingFace TEI Rankings |
| `nim_rankings` | NVIDIA NIM Rankings |
| `huggingface_generate` | HuggingFace TGI API |
| `image_generation` | OpenAI Image Generation API |
| `solido_rag` | SOLIDO RAG API |
| `template` | Custom Jinja2 template |

## Dataset Types

| Type | Description |
|------|-------------|
| `single_turn` | JSONL with single request per line |
| `multi_turn` | JSONL with conversation histories |
| `random_pool` | Pool of prompts for random sampling |
| `mooncake_trace` | Timestamped trace for replay |