# Claude Code Skills Library for AIOps Engineers

A curated collection of Claude Code agents, skills, and commands for building and operating enterprise AI products.

---

## MCP Server Setup

### Kubernetes
```bash
claude mcp add k8s -e KUBECONFIG=~/.kube/config -- npx -y @modelcontextprotocol/server-kubernetes --read-only
```

**Example prompts:**
- `List all pods in llms namespace and their status`
- `Debug pod nginx-abc123 in default namespace. Check status, logs, events, and resource usage`
- `Fix CrashLoopBackOff in pod app-xyz namespace staging. Check previous logs, deployment spec, events; patch resources`
- `List all Helm releases in prod namespace and their status`
- `Troubleshoot why Helm chart nginx failed to deploy. Check deployments, pods, logs, and events`

### Docker
```bash
claude mcp add docker -- npx -y @modelcontextprotocol/server-docker
```

---

## Agents

Specialized Claude agents for each phase of the AI product lifecycle.

**Usage:** `Use <agent-name> to <task>`

### Design & Architecture

| Agent | Model | Description |
|-------|-------|-------------|
| docs-architect | opus | Technical documentation generation |
| tdd-orchestrator | opus | TDD workflow orchestration, test-first development |

### Planning

| Agent | Model | Description |
|-------|-------|-------------|
| kubernetes-architect | opus | K8s/GitOps architecture, EKS/AKS/GKE, service mesh, platform engineering |

### Development

| Agent | Model | Description |
|-------|-------|-------------|
| bash-pro | sonnet | Shell scripting, automation |
| cpp-pro | sonnet | C++ development, performance optimization |
| rust-engineer | sonnet | Rust development, memory safety |
| mcp-developer | sonnet | MCP server development |

### Review & Quality

| Agent | Model | Description |
|-------|-------|-------------|
| code-reviewer | opus | Code quality, security vulnerabilities, performance analysis |
| architect-reviewer | opus | System design validation, architectural patterns, scalability analysis |
| qa-expert | opus | Testing strategies, quality assurance |

### Operations

| Agent | Model | Description |
|-------|-------|-------------|
| debugger | sonnet | Root cause analysis, systematic debugging |
| devops-troubleshooter | sonnet | Infrastructure issue diagnosis |
| refactoring-specialist | sonnet | Code improvement, technical debt reduction |
| git-workflow-manager | sonnet | Git operations, branching strategies |
| prompt-engineer | sonnet | Prompt optimization, LLM tuning |

### Agent-Skill Integration

Recommended skills for each agent to maximize effectiveness.

| Agent | Recommended Skills |
|-------|-------------------|
| docs-architect | `generating-documentation`, `writing-plans`, `langchain-architecture` |
| tdd-orchestrator | `python-testing-patterns`, `writing-plans`, `python-design-patterns` |
| kubernetes-architect | `helm-chart-scaffolding`, `k8s-manifest-generator`, `k8s-security-policies`, `implementing-gitops`, `planning-disaster-recovery` |
| aiops-architect | `helm-chart-scaffolding`, `k8s-manifest-generator`, `helm-chart-vllm`, `aiperf-benchmark` |
| bash-pro | `writing-dockerfiles`, `implementing-gitops` |
| cpp-pro | `high-performance-inference`, `flash-attention`, `debug-cuda-crash` |
| rust-engineer | `high-performance-inference`, `async-python-patterns`, `qdrant` |
| mcp-developer | `langchain-architecture`, `prompt-engineering-patterns`, `python-error-handling` |
| code-reviewer | `python-design-patterns`, `python-testing-patterns`, `python-error-handling`, `k8s-security-policies` |
| architect-reviewer | `llm-serving-patterns`, `implementing-mlops`, `planning-disaster-recovery`, `slo-implementation` |
| qa-expert | `python-testing-patterns`, `evaluating-llms-harness`, `slo-implementation` |
| debugger | `debug-cuda-crash`, `python-error-handling`, `python-testing-patterns` |
| devops-troubleshooter | `operating-kubernetes`, `prometheus-configuration`, `grafana-dashboards`, `debug-cuda-crash`, `implementing-gitops` |
| refactoring-specialist | `python-design-patterns`, `python-testing-patterns`, `async-python-patterns` |
| git-workflow-manager | `implementing-gitops`, `writing-plans` |
| prompt-engineer | `prompt-engineering-patterns`, `langsmith`, `evaluating-llms-harness`, `langchain-architecture` |

---

## Skills

Domain-specific knowledge bases for AI product development.

**Usage:** `/<skill-name> <task>`

### Planning & Design

| Skill | Description |
|-------|-------------|
| brainstorming | Ideation and exploration techniques |
| writing-plans | Implementation planning with TDD |
| notebooklm | Query Google NotebookLM for research |
| planning-disaster-recovery | DR planning and resilience |

### Python Development

| Skill | Description |
|-------|-------------|
| async-python-patterns | Async/await, concurrency patterns |
| python-design-patterns | Design patterns in Python |
| python-error-handling | Exception handling, error recovery |
| python-testing-patterns | pytest, mocking, test strategies |

### LLM Serving & Inference

| Skill | Description |
|-------|-------------|
| llm-serving-patterns | Architecture patterns for LLM APIs |
| vllm | High-throughput LLM serving with PagedAttention |
| serving-llms-vllm | Production vLLM deployment |
| sglang | Structured generation, constrained decoding |
| tensorrt-llm | NVIDIA TensorRT-LLM optimization |
| high-performance-inference | Inference optimization strategies |
| awq | Activation-aware weight quantization |
| flash-attention | Efficient attention mechanisms |
| helm-chart-vllm | Helm chart vLLM deployment |
| aiperf-benchmark | LLM performance benchmarking |

### AI/ML Engineering

| Skill | Description |
|-------|-------------|
| implementing-mlops | End-to-end MLOps: MLflow, feature stores, model serving |
| evaluating-llms-harness | LLM evaluation with lm-evaluation-harness |
| langchain-architecture | LangChain/LangGraph patterns |
| langsmith | LLM observability and tracing |
| prompt-engineering-patterns | Prompt design, few-shot, chain-of-thought |
| qdrant | Vector database operations |
| rag-implementation | RAG systems, semantic search |

### Kubernetes & Infrastructure

| Skill | Description |
|-------|-------------|
| helm-chart-scaffolding | Helm chart development |
| k8s-manifest-generator | Kubernetes manifest generation |
| k8s-security-policies | RBAC, network policies, pod security |
| operating-kubernetes | K8s cluster operations |
| writing-dockerfiles | Dockerfile best practices |

### Monitoring & Observability

| Skill | Description |
|-------|-------------|
| grafana-dashboards | Grafana dashboard design |
| prometheus-configuration | Prometheus setup and alerting |
| slo-implementation | SLO/SLI patterns, error budgets |

### GitOps & Documentation

| Skill | Description |
|-------|-------------|
| implementing-gitops | ArgoCD, Flux, GitOps workflows |
| guiding-users | User guidance and onboarding |
| generating-documentation | Auto-generate technical docs |

### Debugging

| Skill | Description |
|-------|-------------|
| debug-cuda-crash | CUDA debugging, GPU troubleshooting |

---

## Commands

Slash commands for common development tasks.

| Command | Description |
|---------|-------------|
| `/commit` | Create git commits with conventional format |
| `/tech-debt` | Analyze and remediate technical debt |
| `/refactor-clean` | Refactor code for quality and maintainability |
| `/langchain-agent` | Create LangGraph-based agents |
| `/prompt-optimize` | Optimize prompts for production LLMs |

---

## Workflow Examples

### Troubleshoot Production Issues

```
1. use debugger to analyze error logs and stack traces
2. use devops-troubleshooter to check infrastructure
3. /debug-cuda-crash if GPU-related issues
4. use code-reviewer to identify root cause in code
5. /commit fix with conventional commit message
```

### End-to-End LLM Deployment with K8s & Helm

A comprehensive workflow from ideation to production deployment of an LLM serving infrastructure.

**Phase 1: Research & Ideation**
1. /notebooklm query research notebooks for a topic

Example:

```
/notebooklm query research notebooks vLLM with LMCache from my notebook https://notebooklm.google.com/notebook/9bd1386c-af79-4407-a10e-27f9c0951e9c?pli=1 and save to my global memory ~/.claude/plans/lmcache-memory.md

Please analyze and synthesize information about:

1. vLLM + LMCache Integration
   - Architecture and how LMCache works with vLLM
   - Key configuration parameters for optimal performance
   - Setup and deployment considerations

2. Memory Management Best Practices
   - GPU memory optimization techniques
   - KV cache management strategies
   - Memory pooling and allocation patterns
   - Trade-offs between memory usage and throughput

3. Serving Optimization
   - Batch size tuning recommendations
   - Request queuing and scheduling strategies
   - Latency vs throughput optimization
   - Multi-GPU and distributed serving considerations

4. LMCache-Specific Features
   - Cache hit rate optimization
   - Cache eviction policies
   - Prefix caching benefits and use cases
   - Performance benchmarks and metrics

5. Production Deployment
   - Common pitfalls and how to avoid them
   - Monitoring and observability recommendations
   - Resource sizing guidelines
   - Cost optimization strategies

Please provide specific recommendations, configuration examples, and any relevant performance data or case studies found in the research notebooks.
```
2. /brainstorming explore deployment requirements and constraints

Example:

**Phase 2: Architecture & Design**
3. use docs-architect to create system design documentation
4. use architect-reviewer to validate architecture decisions

Example:
```
Use docs-architect agent reate comprehensive system design documentation for vLLM integration with LMCache follow my codebase. Include:
1. Architecture overview with component interactions
2. KV cache storage and retrieval flow
3. Integration points between vLLM and LMCache
4. Configuration options and their impact
5. Performance considerations
6. Deployment patterns
7. Sequence diagrams for key operations
Output to docs/lmcache-system-design.md

Use architect-reviewer agent to review the system design documentation at @docs/lmcache-system-design.md follow my codebase for:
- Technical accuracy and completeness
- Clarity of architectural decisions
- Proper coverage of integration points
- Missing edge cases or failure modes
- Consistency with codebase implementation
- Recommendations for improvements
Provide detailed feedback and suggestions.
```

**Phase 3: Planning**
5. /writing-plans create implementation plan with TDD approach
6. /planning-disaster-recovery define RTO/RPO and backup strategies

Example:
```
/writing-plans Create a detailed Implementation Plan into docs/plan/lmcache-integration-planner.md from my System Design Document below. Include:

- Implementation phases with 1 sprint (2 weeks)
- Task breakdown with priorities and effort estimates
- Risk assessment and mitigation strategies
- Testing and deployment strategy
- Resource and infrastructure requirements
- Key milestones and deliverables

Ensure the implementation plan is actionable, realistic, and covers all components from the system design.

System Design Document: docs/lmcache-system-design.md
```

**Phase 4: Implementation**

7. /vllm configure model serving with tensor parallelism
8. /high-performance-inference optimize with AWQ quantization
9. /k8s-manifest-generator create Deployment, Service, ConfigMap
10. /helm-chart-scaffolding or /helm-chart-vllm package as reusable Helm chart

Example:
```
Use aiops-architect agent to implement my plan from document requirements @docs/plan/lmcache-integration-planner.md
```

**Phase 5: Review & Quality**
```
11. use code-reviewer agent to analyze security and performance
12. use refactoring-specialist agent if code improvements needed
```

**Phase 6: Benchmarking & Debugging**
```
13. Deploy to staging cluster with helm install
14. use devops-troubleshooter if pod issues occur
15. use debugger for application-level errors
16. /debug-cuda-crash if GPU-related issues
17. /aiperf-benchmark to benchmark performance of model
```

**Phase 7: Documentation**
```
18. /generating-documentation create deployment runbook and API docs
19. /commit document changes with conventional commit
```