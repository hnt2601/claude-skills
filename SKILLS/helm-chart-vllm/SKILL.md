---
name: helm-chart-vllm
description: Create vLLM Helm chart deployments compatible with ai-models-service architecture. Use when adding new language/vision-language models, configuring multi-GPU deployments, or optimizing vLLM serving performance on Kubernetes.
version: 1.1.0
tags: [vLLM, Helm, Kubernetes, GPU, LLM Serving, Tensor Parallelism, FP8 Quantization]
author: DevOps Team
last_updated: 2025-02-02
prerequisites:
  - Helm v3.10+
  - kubectl configured with cluster access
  - Access to GPU-enabled nodes
  - Write access to ai-models-service repository
---

# Helm Chart vLLM Deployments

Comprehensive guidance for creating vLLM model deployments using Helm charts in the ai-models-service repository.

## Table of Contents

- [Purpose](#purpose)
- [When to Use This Skill](#when-to-use-this-skill)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Repository Architecture Overview](#repository-architecture-overview)
- [Step-by-Step Workflow](#step-by-step-workflow-adding-a-new-model)
- [Choosing the Right Template](#choosing-the-right-template)
- [Configuration Templates](#configuration-templates)
- [Host Path Storage Configuration](#host-path-storage-configuration)
- [Performance Optimization Settings](#performance-optimization-settings)
- [Security Configuration](#security-configuration)
- [GPU Topology and Scheduling](#gpu-topology-and-scheduling)
- [Graceful Shutdown and Updates](#graceful-shutdown-and-zero-downtime-updates)
- [Observability and Monitoring](#observability-and-monitoring)
- [Network Security](#network-security)
- [Namespace Resource Management](#namespace-resource-management)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)
- [Validation Commands](#validation-commands)
- [Glossary](#glossary)

## Purpose

This skill provides step-by-step instructions for deploying vLLM-based models (language and vision-language) using the repository's Helm chart architecture, including multi-GPU configurations, quantization settings, and performance optimizations.

## When to Use This Skill

Use this skill when you need to:

- Add a new language or vision-language model to the cluster
- Configure multi-GPU tensor parallelism for large models
- Set up MoE (Mixture of Experts) models like DeepSeek or Qwen3-Coder
- Enable function calling or tool use capabilities
- Optimize inference performance (prefix caching, chunked prefill)
- Troubleshoot vLLM deployment issues

## Prerequisites

Before using this skill, ensure you have:

- **Helm v3.10+** installed (`helm version`)
- **kubectl** configured with cluster access (`kubectl cluster-info`)
- **Access to GPU-enabled nodes** with NVIDIA device plugin
- **Write access** to the ai-models-service repository
- **Understanding** of your model's GPU requirements (VRAM, tensor parallelism)

**Verify prerequisites:**

```bash
# Check Helm version
helm version --short

# Check kubectl access
kubectl get nodes -o custom-columns="NODE:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu"

# Check namespace access
kubectl auth can-i create deployments -n llms
```

## Quick Start

Deploy a single-GPU model in 5 minutes:

```bash
# 1. Check available NodePorts
grep -r "nodePort:" environments/staging/language/ | grep -oE '300[0-9]{2}' | sort -n | tail -5

# 2. Create model config (copy from existing model)
cp environments/staging/language/qwen3-32b.yaml environments/staging/language/my-new-model.yaml

# 3. Edit the config (update fullnameOverride, model path, servedModelName, nodePort)
vim environments/staging/language/my-new-model.yaml

# 4. Enable in values.yaml
# Add: language.models.my-new-model.enabled: true

# 5. Add to helmfile.yaml.gotmpl and deploy
./scripts/cli.sh deploy diff -e staging
./scripts/cli.sh deploy sync -e staging
```

## Repository Architecture Overview

### Umbrella Chart Structure

```
ai-models-umbrella/
├── helmfile.yaml.gotmpl          # Unified helmfile for all environments
├── Chart.yaml                    # Umbrella chart with aliases
├── subcharts/
│   └── vllm-models/              # vLLM sub-chart (language + vision-language)
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values.schema.json
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── poddisruptionbudget.yaml
│           └── _helpers.tpl
└── environments/
    ├── staging/
    │   ├── values.yaml           # Enable/disable models
    │   ├── language/*.yaml       # Language model configs
    │   └── vision-language/*.yaml
    ├── production
    │   ├── values.yaml           # Enable/disable models
    │   ├── language/*.yaml       # Language model configs
    │   └── vision-language/*.yaml
```

### Three-Level Configuration Hierarchy

1. **Sub-chart defaults** (`subcharts/vllm-models/values.yaml`) - Base structure
2. **Environment values** (`environments/<env>/values.yaml`) - Global settings, enable flags
3. **Model configs** (`environments/<env>/language/*.yaml`) - Model-specific overrides

### Two Deployment Methods

| Method | Use Case | Values Pattern |
|--------|----------|----------------|
| **OCI Chart** | CI/CD pipelines | `<alias>.hps.models.path` |
| **Helmfile** | Local development | `global.hps.models.path` |

### Naming Convention

All model deployments follow the pattern: `llms-v2-<model-name>`

- `llms` - Namespace identifier
- `v2` - Current deployment architecture version
- `<model-name>` - Lowercase model identifier with hyphens

**Note:** In this document, `<placeholder>` indicates values you must replace:
- `<model-name>` - Your model's identifier (e.g., `qwen3-32b`)
- `<env>` - Environment name (`staging`, `production-vn`, `production-jp`)
- `<port>` - Assigned NodePort number

## Step-by-Step Workflow: Adding a New Model

### Step 1: Check NodePort Availability

**Port allocation ranges:**

| Range | Category |
|-------|----------|
| 30006-30013 | Language Models |
| 30014-30017 | Embedding Models |
| 30018-30022 | Vision-Language Models |
| 30023-30024 | Text-to-Speech |
| 30025-30035 | Language Models (Extended) |
| 30100-30199 | Speech-to-Text |
| 30500-30599 | Custom/FAI Models |

**Check for conflicts:**

```bash
# Find duplicate ports (returns duplicates if any exist)
grep -r "nodePort:" environments/staging/ 2>/dev/null | grep -oE '300[0-9]{2}' | sort | uniq -d

# Expected output: (empty if no conflicts)

# Find next available port
grep -r "nodePort:" environments/staging/ | grep -oE '300[0-9]{2}' | sort -n | tail -5
```

### Step 2: Create Model Configuration File

**File location:**
- Language models: `environments/<env>/language/<model-name>.yaml`
- Vision-language: `environments/<env>/vision-language/<model-name>.yaml`

**Required fields:**

```yaml
# Naming (REQUIRED)
fullnameOverride: llms-v2-<model-name>   # Kubernetes resource name
modelName: <model-name>                   # Label for identification

# Image (REQUIRED)
image:
  tag: "v0.10.2"  # vLLM version

# Deployment (REQUIRED)
deployment:
  model: /models/<org>/<model-path>       # Path inside container
  servedModelName: <model-display-name>   # Exposed in /v1/models endpoint
  tensorParallel: 1                       # Number of GPUs for model sharding
  numGPUs: 1                              # Must match tensorParallel
  gpuMemoryUtilization: 0.85              # 85% VRAM usage (safe default)
  enableChunkedPrefill: true              # Reduces first-token latency

# Service (REQUIRED)
service:
  nodePort: <assigned-port>               # From port allocation ranges

# Resources (REQUIRED) - Use Guaranteed QoS (requests == limits)
resources:
  limits:
    nvidia.com/gpu: 1
    memory: 200Gi
    cpu: "16"
  requests:
    nvidia.com/gpu: 1
    memory: 200Gi   # Match limits for Guaranteed QoS
    cpu: "16"       # Match limits for Guaranteed QoS
```

### Step 3: Enable Model in Environment Values

Edit `environments/<env>/values.yaml`:

```yaml
language:
  enabled: true
  models:
    <model-name>:
      enabled: true
```

### Step 4: Add to Helmfile

Add release to `helmfile.yaml.gotmpl`:

```yaml
- name: <model-name>
  namespace: llms
  chart: ./subcharts/vllm-models
  values:
    - environments/{{ .Environment.Name }}/language/<model-name>.yaml
  installed: {{ and (.Values | get "language.enabled" false) (.Values | get "language.models.<model-name>.enabled" false) }}
```

### Step 5: Validate & Test

```bash
# Lint and verify
./scripts/cli.sh chart verify

# Test template rendering
helm template test-release ./subcharts/vllm-models \
  -f environments/staging/language/<model-name>.yaml

# Preview deployment
./scripts/cli.sh deploy diff -e staging

# Deploy
./scripts/cli.sh deploy sync -e staging

# Verify deployment
kubectl get pods -n llms -l model=<model-name>

# Test API endpoint
curl http://<node-ip>:<nodePort>/v1/models
# Expected: {"data":[{"id":"<servedModelName>","object":"model",...}]}
```

## Choosing the Right Template

| Your Situation | Use Template |
|----------------|--------------|
| First deployment, small model (<40B) | Template 1: Single GPU |
| Large model (70B+) | Template 2: Multi-GPU |
| Image/video processing | Template 3: Vision-Language |
| MoE architecture (DeepSeek, Qwen3-MoE) | Template 4: MoE |
| Need tool calling/function use | Template 5: Function Calling |
| Memory-constrained environment | Template 6: LMCache |

## Configuration Templates

### Template 1: Single GPU Model (Standard)

Use for 7B-32B models on single GPU.

```yaml
fullnameOverride: llms-v2-qwen3-32b
modelName: qwen3-32b

image:
  tag: "v0.10.2"

deployment:
  model: /models/Qwen/Qwen3-32B-FP8
  servedModelName: Qwen3-32B
  tensorParallel: 1
  numGPUs: 1
  gpuMemoryUtilization: 0.85
  enableChunkedPrefill: true
  quantization: fp8
  # Graceful shutdown
  terminationGracePeriodSeconds: 60

prefixCaching:
  enable: true

service:
  nodePort: 30025

# Guaranteed QoS: requests == limits
resources:
  limits:
    nvidia.com/gpu: 1
    memory: 200Gi
    cpu: "16"
  requests:
    nvidia.com/gpu: 1
    memory: 200Gi
    cpu: "16"

probes:
  startup:
    failureThreshold: 120  # 10 minutes max load time
  readiness:
    periodSeconds: 10
    failureThreshold: 3
  liveness:
    initialDelaySeconds: 300
    periodSeconds: 30
```

### Template 2: Multi-GPU Model (Tensor Parallelism)

Use for 70B+ models requiring multiple GPUs.

**Critical:** `tensorParallel` MUST equal `numGPUs`

```yaml
fullnameOverride: llms-v2-llama-3-3-70b-instruct
modelName: llama-3-3-70b-instruct

image:
  tag: "v0.10.2"

deployment:
  model: /models/meta-llama/Llama-3.3-70B-Instruct-FP8
  servedModelName: Llama-3.3-70B-Instruct
  tensorParallel: 4
  numGPUs: 4
  gpuMemoryUtilization: 0.9
  enableChunkedPrefill: true
  quantization: fp8
  terminationGracePeriodSeconds: 120

# Shared memory for NCCL communication
shmSize: 32Gi  # Scale: TP=2: 16Gi, TP=4: 32Gi, TP=8: 64Gi+

# NCCL optimization for multi-GPU
nccl:
  gpuMem: true
  logLevel: "WARN"

prefixCaching:
  enable: true

service:
  nodePort: 30030

# Guaranteed QoS for multi-GPU
resources:
  limits:
    nvidia.com/gpu: 4
    memory: 400Gi
    cpu: "32"
  requests:
    nvidia.com/gpu: 4
    memory: 400Gi
    cpu: "32"

probes:
  startup:
    failureThreshold: 240  # 20 minutes for large model
  readiness:
    periodSeconds: 10
  liveness:
    initialDelaySeconds: 600
    periodSeconds: 30

# Pod Disruption Budget for HA
podDisruptionBudget:
  enabled: true
  minAvailable: 1
```

### Template 3: Vision-Language Model

Use for multimodal models (Gemma-3, Qwen2-VL, etc.).

```yaml
fullnameOverride: llms-v2-gemma-3-27b-it
modelName: gemma-3-27b-it

image:
  tag: "v0.10.2"

deployment:
  model: /models/google/gemma-3-27b-it
  servedModelName: Gemma-3-27B-IT
  tensorParallel: 1
  numGPUs: 1
  gpuMemoryUtilization: 0.95
  enableChunkedPrefill: true
  terminationGracePeriodSeconds: 60

  # Vision-specific settings
  visionModel: true
  limitMmPerPrompt: '{"image": 12}'       # Max 12 images per prompt
  disableMmPreprocessorCache: false       # Cache preprocessed images
  attentionBackend: "FLASH_ATTN"          # Flash attention for vision

prefixCaching:
  enable: true

service:
  nodePort: 30027

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 64Gi
    cpu: "16"
  requests:
    nvidia.com/gpu: 1
    memory: 64Gi
    cpu: "16"

probes:
  startup:
    failureThreshold: 120
```

### Template 4: MoE Model (Expert Parallelism)

Use for Mixture-of-Experts models (DeepSeek, Qwen3-MoE).

```yaml
fullnameOverride: llms-v2-deepseek-r1
modelName: deepseek-r1

image:
  tag: "v0.10.2"

deployment:
  model: /models/deepseek-ai/DeepSeek-R1-FP8
  servedModelName: DeepSeek-R1
  tensorParallel: 8
  numGPUs: 8
  gpuMemoryUtilization: 0.95
  enableChunkedPrefill: true
  terminationGracePeriodSeconds: 180

  # MoE-specific settings
  enableExpertParallel: true
  asyncScheduling: true

  # Function calling
  functionCall: true
  toolCallParser: "deepseek"

# Large shared memory for MoE communication
shmSize: 784Gi

# NCCL optimization
nccl:
  gpuMem: true
  logLevel: "WARN"

prefixCaching:
  enable: true

service:
  nodePort: 30038

# Priority for critical models
priorityClassName: gpu-critical

resources:
  limits:
    nvidia.com/gpu: 8
    memory: 768Gi
    cpu: "64"
  requests:
    nvidia.com/gpu: 8
    memory: 768Gi
    cpu: "64"

probes:
  startup:
    failureThreshold: 480  # 40 minutes for massive model
  liveness:
    initialDelaySeconds: 900
    periodSeconds: 60
```

### Template 5: Function Calling Model

Use for models with tool use capabilities.

```yaml
fullnameOverride: llms-v2-qwen3-coder-480b
modelName: qwen3-coder-480b

image:
  tag: "v0.10.2"

deployment:
  model: /models/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
  servedModelName: Qwen3-Coder-480B
  tensorParallel: 4
  numGPUs: 4
  gpuMemoryUtilization: 0.9
  enableChunkedPrefill: true
  terminationGracePeriodSeconds: 120

  # Function calling settings
  functionCall: true
  toolCallParser: "qwen3_coder"  # Options: qwen3_coder, deepseek, hermes

  # Expert parallelism for MoE
  enableExpertParallel: true
  asyncScheduling: true

# Custom environment variables
extraEnv:
  - name: VLLM_USE_DEEP_GEMM
    value: "1"
  - name: VLLM_MOE_USE_DEEP_GEMM
    value: "0"

nccl:
  gpuMem: true

shmSize: 32Gi

service:
  nodePort: 30032

resources:
  limits:
    nvidia.com/gpu: 4
    memory: 400Gi
    cpu: "32"
  requests:
    nvidia.com/gpu: 4
    memory: 400Gi
    cpu: "32"
```

### Template 6: LMCache KV Offloading

Use when GPU memory is constrained and need KV cache offloading.

```yaml
fullnameOverride: llms-v2-qwen3-32b-lmcache
modelName: qwen3-32b-lmcache

image:
  tag: "v0.10.2"

deployment:
  model: /models/Qwen/Qwen3-32B-FP8
  servedModelName: Qwen3-32B-LMCache
  tensorParallel: 1
  numGPUs: 1
  gpuMemoryUtilization: 0.85
  enableChunkedPrefill: true
  terminationGracePeriodSeconds: 60

  # KV Offloading configuration
  kvOffloading:
    enabled: true
    backend: "lmcache"
    cpuOffloadingBufferSize: 4  # GB
    mmProcessorCacheGb: 4
    mmProcessorCacheType: "lru"

extraEnv:
  - name: LMCACHE_CHUNK_SIZE
    value: "512"
  - name: LMCACHE_LOG_LEVEL
    value: "INFO"
  - name: VLLM_USE_V1
    value: "1"

service:
  nodePort: 30040

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 300Gi  # Extra for CPU offloading
    cpu: "24"
  requests:
    nvidia.com/gpu: 1
    memory: 300Gi
    cpu: "24"
```

## Host Path Storage Configuration

### Storage Paths

| Path | Content | Usage |
|------|---------|-------|
| `/var/modas-models/fp8_models/` | FP8 quantized models | Most language/vision models |
| `/var/modas-models/fp16_models/` | FP16 models | Embedding, rerank, some STT |

### Template Logic

The deployment template uses conditional path resolution:

```yaml
# In sub-chart template
volumes:
  - name: model-dir
    hostPath:
      {{- if .Values.hps }}
      path: {{ .Values.hps.models.path | default .Values.global.hps.models.path }}
      {{- else }}
      path: {{ .Values.global.hps.models.path }}
      {{- end }}
```

**Example rendered output:**

For Helmfile deployment with `global.hps.models.path: /var/modas-models/fp8_models/`:
```yaml
volumes:
  - name: model-dir
    hostPath:
      path: /var/modas-models/fp8_models/
```

## Performance Optimization Settings

### GPU Memory Utilization

| Model Type | Recommended | Notes |
|------------|-------------|-------|
| Standard LLM | 0.85 | Leaves headroom for kernel buffers |
| High throughput | 0.90-0.95 | For MoE models, production |
| Development | 0.70-0.80 | Extra safety margin |

### Prefix Caching

Enable for repeated prompt patterns:

```yaml
prefixCaching:
  enable: true
```

Benefits:
- Reduces TTFT for similar prompts
- Saves GPU memory on common prefixes
- Essential for chatbot applications

### Chunked Prefill

Enable for long prompts:

```yaml
deployment:
  enableChunkedPrefill: true
```

Benefits:
- Reduces first-token latency on long inputs
- Better memory management
- Enabled by default for all models

### Shared Memory Scaling

| Tensor Parallel | Recommended shmSize |
|-----------------|---------------------|
| TP=1 | 16Gi (default) |
| TP=2 | 16Gi |
| TP=4 | 32Gi |
| TP=8 | 64Gi |
| TP=8 + MoE | 784Gi (for 600B+ models) |

### Resource Sizing Guidelines

| Model Size | Memory Limit | Memory Request | CPU Limit | CPU Request |
|------------|--------------|----------------|-----------|-------------|
| < 10B params | 64Gi | 64Gi | 16 | 16 |
| 10-40B params | 200Gi | 200Gi | 16 | 16 |
| 40-100B params | 400Gi | 400Gi | 32 | 32 |
| > 100B params | 768Gi | 768Gi | 64 | 64 |

**Important:** For Guaranteed QoS (recommended for GPU workloads), requests MUST equal limits.

## Security Configuration

### Pod Security Context

While vLLM containers require GPU access, security should still be enforced where possible.

**Add to model configuration:**

```yaml
# Security context for production deployments
securityContext:
  runAsNonRoot: false  # vLLM may need root for GPU access
  seccompProfile:
    type: RuntimeDefault

containerSecurityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
    add:
      - SYS_PTRACE  # Required for NCCL debugging (optional)
  readOnlyRootFilesystem: false  # vLLM needs write access
```

**Note:** GPU workloads have unique security requirements:
- NVIDIA device plugin requires access to GPU devices
- Cannot use `readOnlyRootFilesystem: true` due to HuggingFace cache writes
- Use `RuntimeDefault` seccomp profile as minimum

### Pod Security Standards Compliance

For namespaces with Pod Security Standards enabled, vLLM pods require **baseline** or **privileged** level:

```yaml
# Namespace labels for Pod Security Admission
apiVersion: v1
kind: Namespace
metadata:
  name: llms
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

## GPU Topology and Scheduling

### NVIDIA GPU Topology Awareness

For multi-GPU deployments (tensorParallel > 1), proper GPU topology is critical for performance.

**Node Selector for NVLink-connected GPUs:**

```yaml
global:
  nodeSelector:
    nvidia.com/gpu.product: "NVIDIA-A100-SXM4-80GB"  # NVLink-capable
    nvidia.com/gpu.count: "8"
```

**Topology Spread for Multi-Replica:**

```yaml
deployment:
  topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: kubernetes.io/hostname
      whenUnsatisfiable: DoNotSchedule
      labelSelector:
        matchLabels:
          model: {{ .Values.modelName }}
```

### NCCL Topology Optimization

For optimal multi-GPU communication:

```yaml
deployment:
  nccl:
    gpuMem: true      # Enable NCCL GPU direct memory
    logLevel: "WARN"  # Set to INFO for debugging

  extraEnv:
    - name: NCCL_IB_DISABLE
      value: "0"  # Enable InfiniBand if available
    - name: NCCL_P2P_LEVEL
      value: "NVL"  # Force NVLink when available
```

## Graceful Shutdown and Zero-Downtime Updates

### Termination Grace Period

LLM inference requests can be long-running. Configure adequate shutdown time:

```yaml
deployment:
  terminationGracePeriodSeconds: 120  # 2 minutes for request completion

  # Rolling update strategy for zero-downtime
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0   # Never remove existing pod before new one is ready
      maxSurge: 1         # Create one new pod before terminating old
```

### Termination Grace Period Guidelines

| Model Size | terminationGracePeriodSeconds |
|------------|-------------------------------|
| < 30B | 60 |
| 30B-70B | 120 |
| > 70B | 180 |
| MoE (600B+) | 300 |

### Probe Execution Order

1. **startupProbe** runs first until success (or failureThreshold reached)
2. Only after startupProbe succeeds, **livenessProbe** and **readinessProbe** begin
3. readinessProbe controls Service endpoint registration
4. livenessProbe triggers container restart on failure

### Model-Size Based Startup Thresholds

| Model Size | Quantization | Approximate Load Time | failureThreshold |
|------------|-------------|----------------------|------------------|
| 7B-13B | FP8 | 1-3 minutes | 60 |
| 32B | FP8 | 3-5 minutes | 120 |
| 70B | FP8 | 5-10 minutes | 180 |
| 405B | FP8 | 15-25 minutes | 360 |
| 600B+ MoE | FP8 | 20-40 minutes | 480 |

## Observability and Monitoring

### Prometheus Metrics

vLLM exposes Prometheus metrics at `/metrics`:

```yaml
deployment:
  podAnnotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `vllm:num_requests_running` | Current in-flight requests | > 80% of max_num_seqs |
| `vllm:num_requests_waiting` | Queued requests | > 100 for > 5 minutes |
| `vllm:gpu_cache_usage_perc` | KV cache utilization | > 95% |
| `vllm:avg_generation_throughput_toks_per_s` | Generation speed | < baseline - 20% |
| `vllm:time_to_first_token_seconds` | TTFT latency | > SLA threshold |
| `vllm:time_per_output_token_seconds` | Per-token latency | > SLA threshold |

### ServiceMonitor for Prometheus Operator

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-models
  namespace: llms
spec:
  selector:
    matchLabels:
      app: vllm-multi
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
  namespaceSelector:
    matchNames:
      - llms
```

### Logging Configuration

```yaml
global:
  logging:
    level: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

deployment:
  extraEnv:
    - name: VLLM_LOGGING_LEVEL
      value: "INFO"
```

## Network Security

### Network Policy for vLLM Pods

Restrict network access to only required endpoints:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vllm-network-policy
  namespace: llms
spec:
  podSelector:
    matchLabels:
      app: vllm-multi
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow traffic from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
    # Allow traffic from monitoring namespace
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 8000
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
    # Allow NCCL communication between pods (same namespace)
    - to:
        - podSelector:
            matchLabels:
              app: vllm-multi
```

**Note:** Network policies require a CNI that supports them (Calico, Cilium, etc.).

## Namespace Resource Management

### Priority Classes for GPU Workloads

Create priority classes to ensure critical models get scheduled first:

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-critical
value: 1000000
globalDefault: false
description: "Critical GPU workloads - production models"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-high
value: 100000
description: "High priority GPU workloads"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-normal
value: 10000
description: "Normal GPU workloads - development/staging"
```

**Use in model configuration:**

```yaml
deployment:
  priorityClassName: gpu-critical  # For production models
```

### Resource Quota for GPU Namespace

Prevent resource exhaustion:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: llms
spec:
  hard:
    requests.nvidia.com/gpu: "32"  # Max 32 GPUs in namespace
    limits.nvidia.com/gpu: "32"
    requests.memory: "2Ti"
    limits.memory: "4Ti"
    pods: "50"
```

## Cost Optimization

### Right-Sizing GPU Resources

| Model Size | Minimum GPU | Recommended GPU |
|------------|-------------|-----------------|
| 7B-13B | 1x 24GB (L4) | 1x 40GB (A100) |
| 32B FP8 | 1x 80GB | 1x 80GB |
| 70B FP8 | 2x 80GB | 4x 80GB |
| 405B FP8 | 8x 80GB | 8x 80GB |

### Spot/Preemptible Instances

For non-critical workloads:

```yaml
global:
  nodeSelector:
    cloud.google.com/gke-spot: "true"
    # OR for AWS:
    # eks.amazonaws.com/capacityType: SPOT

deployment:
  terminationGracePeriodSeconds: 30  # Spot termination notice is 30s
```

## Troubleshooting

### Issue: Out of Memory During Model Loading

**Symptoms:** Pod crashes with OOM, GPU memory exhausted

**Solutions:**

1. Reduce GPU memory utilization:
```yaml
deployment:
  gpuMemoryUtilization: 0.7
```

2. Use quantization:
```yaml
deployment:
  quantization: fp8  # or awq, gptq
```

3. Increase tensor parallelism:
```yaml
deployment:
  tensorParallel: 2
  numGPUs: 2
```

### Issue: Slow Model Startup (>15 minutes)

**Symptoms:** Pod stays in `Running` but not ready

**Solutions:**

1. Extend startup probe:
```yaml
probes:
  startup:
    failureThreshold: 360  # 30 minutes
```

2. Check model path is correct:
```bash
kubectl exec -it <pod> -n llms -- ls -la /models/
```

3. Verify host path storage:
```bash
kubectl describe pod <pod> -n llms | grep -A5 "Volumes:"
```

### Issue: NodePort Conflict

**Symptoms:** Service creation fails

**Solution:**

```bash
# Find conflicting port
kubectl get svc -n llms -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.ports[*].nodePort}{"\n"}{end}' | grep <port>

# Choose unused port from appropriate range
```

### Issue: Multi-GPU Synchronization Failures

**Symptoms:** NCCL errors, GPU communication timeout

**Diagnosis:**

```bash
# Check GPU topology on node
nvidia-smi topo -m

# Check pod's GPU assignment
kubectl exec -it <pod> -n llms -- nvidia-smi

# Verify NCCL topology detection
kubectl logs <pod> -n llms | grep -i nccl
```

**Solutions:**

1. Increase shared memory:
```yaml
shmSize: 64Gi  # or higher
```

2. Enable NCCL GPU memory:
```yaml
nccl:
  gpuMem: true
```

3. Set NCCL debug logging:
```yaml
nccl:
  logLevel: "INFO"
```

### Issue: Low Throughput

**Symptoms:** < 50 requests/second

**Solutions:**

1. Increase max concurrent sequences:
```yaml
deployment:
  maxNumSeqs: 512
```

2. Enable prefix caching:
```yaml
prefixCaching:
  enable: true
```

3. Verify GPU utilization:
```bash
kubectl exec -it <pod> -n llms -- nvidia-smi
```

### Issue: Pod Stuck in Pending (GPU Scheduling)

**Symptoms:** Pod shows `Pending` with insufficient GPU message

**Diagnosis:**

```bash
# Check cluster GPU capacity
kubectl get nodes -o custom-columns="NODE:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu"

# Check GPU allocation
kubectl describe nodes | grep -A5 "nvidia.com/gpu"

# Check pending pods
kubectl get pods -n llms -o wide | grep Pending
```

**Solutions:**

1. Check for GPU fragmentation
2. Review priority classes - higher priority pods preempt lower
3. Scale cluster or deallocate unused GPU workloads

### Issue: Inference Latency Spikes

**Symptoms:** Periodic latency increases, TTFT or TPS degradation

**Diagnosis:**

```bash
# Check KV cache utilization
kubectl exec -it <pod> -n llms -- curl -s localhost:8000/metrics | grep cache

# Check request queue
kubectl exec -it <pod> -n llms -- curl -s localhost:8000/metrics | grep requests

# Check GPU memory
kubectl exec -it <pod> -n llms -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Common Causes:**
- KV cache eviction due to memory pressure
- Request queueing from batch size limits
- Prefix cache misses
- GPU memory fragmentation

**Solutions:**

1. Reduce `gpuMemoryUtilization` to leave headroom
2. Increase `maxNumSeqs` for higher concurrency
3. Enable prefix caching for repeated prompts

### Rollback Procedure

If deployment fails:

```bash
# Check release history
helm history <release-name> -n llms

# Rollback to previous version
helm rollback <release-name> <revision> -n llms

# Or with helmfile
./scripts/cli.sh deploy sync -e staging  # After reverting config files
```

## Validation Commands

```bash
# Lint chart
helm lint ./subcharts/vllm-models

# Test template with values
helm template test ./subcharts/vllm-models \
  -f environments/staging/language/<model>.yaml \
  --debug

# Diff before deploy
./scripts/cli.sh deploy diff -e staging

# Check pod status
kubectl get pods -n llms -l app=vllm-multi

# View logs
./scripts/cli.sh k8s logs <release-name>

# Check GPU allocation
kubectl describe node | grep -A5 "nvidia.com/gpu"

# Test model endpoint
curl http://<node-ip>:<nodePort>/v1/models
curl http://<node-ip>:<nodePort>/health
```

## Quick Reference Tables

### Common vLLM Arguments

| Argument | Values Setting | Description |
|----------|----------------|-------------|
| `--model` | `deployment.model` | Model path in container |
| `--served-model-name` | `deployment.servedModelName` | API model name |
| `--tensor-parallel-size` | `deployment.tensorParallel` | GPU parallelism |
| `--gpu-memory-utilization` | `deployment.gpuMemoryUtilization` | VRAM usage (0-1) |
| `--enable-prefix-caching` | `prefixCaching.enable` | KV cache reuse |
| `--enable-chunked-prefill` | `deployment.enableChunkedPrefill` | Long prompt handling |
| `--quantization` | `deployment.quantization` | awq, gptq, fp8 |
| `--trust-remote-code` | Always enabled | Custom model support |

### Model Category to Chart Mapping

| Model Type | Sub-Chart | Directory |
|------------|-----------|-----------|
| Language models | `vllm-models` | `language/` |
| Vision-language | `vllm-models` | `vision-language/` |
| Embedding | `infinity-models` | `embedding/` |
| Rerank | `infinity-models` | `rerank/` |
| Speech-to-text | `vllm-audio` | `stt/` |
| Text-to-speech | `fptai-tts` | `tts/` |

### Tool Call Parsers

| Parser | Models |
|--------|--------|
| `qwen3_coder` | Qwen3-Coder series |
| `deepseek` | DeepSeek-R1, DeepSeek-V3 |
| `hermes` | Hermes fine-tunes |

## Glossary

| Term | Definition |
|------|------------|
| **TTFT** | Time To First Token - latency before first response token |
| **TPS** | Tokens Per Second - throughput metric |
| **NCCL** | NVIDIA Collective Communications Library - GPU communication |
| **MoE** | Mixture of Experts - sparse model architecture with multiple expert networks |
| **TP** | Tensor Parallelism - splitting model layers across GPUs |
| **FP8** | 8-bit floating point - quantization format for reduced memory |
| **KV Cache** | Key-Value Cache - stores attention states for efficient generation |
| **QoS** | Quality of Service - Kubernetes resource guarantee levels |
| **PDB** | Pod Disruption Budget - protects pods during maintenance |
| **SHM** | Shared Memory - inter-process communication memory (emptyDir) |

## Related Files

- `subcharts/vllm-models/templates/deployment.yaml` - Main deployment template
- `subcharts/vllm-models/values.schema.json` - Values validation schema
- `docs/NODEPORT-MANAGEMENT.md` - Port allocation documentation
- `helmfile.yaml.gotmpl` - Helmfile release definitions

## Related Skills

- `helm-chart-scaffolding` - General Helm chart patterns
- `vllm` - vLLM serving fundamentals
- `operating-kubernetes` - K8s operations
- `k8s-manifest-generator` - Raw K8s manifest creation
