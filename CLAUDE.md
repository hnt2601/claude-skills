# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Claude Code skills library** - a collection of reusable agents, commands, rules, and domain-specific skills for AIOps engineers building and operating enterprise AI products. The repository contains no executable code; it's a reference library of markdown-based configurations and prompts.

## Directory Structure

```
├── AGENTS/      # Agent definitions (19 agents)
├── COMMANDS/    # Slash command templates (5 commands)
├── RULES/       # Behavioral guidelines (git-workflow, performance)
├── SKILLS/      # Domain-specific knowledge bases (42 skills)
├── CLAUDE-ROUTER.md  # Multi-provider model switching guide
├── CLAUDE-CLI.md     # CLI documentation
```

## File Formats

### AGENTS (`AGENTS/*.md`)
```yaml
---
name: agent-name
description: Brief description for agent selection
model: opus | sonnet | haiku
permissionMode: dontAsk, plan  # Optional: control permission prompts
skills:                        # Optional: recommended skills for this agent
  - skill-name-1
  - skill-name-2
---
```

### COMMANDS (`COMMANDS/*.md`)
```yaml
---
allowed-tools: Bash(git add:*), Bash(git commit:*)  # Optional
description: "Command description"
argument-hint: "<required-arg> [optional-arg]"       # Optional
---
```
Dynamic context: `!`backtick commands` inject runtime values (e.g., `!`git status`)

### SKILLS (`SKILLS/<skill-name>/`)
```yaml
---
name: skill-name
description: When to use this skill
version: 1.0.0                 # Optional
tags: [tag1, tag2]             # Optional
dependencies: [dep1, dep2]     # Optional
---
```
Subdirectory structure: `references/`, `examples/`, `assets/`, `scripts/`, `checklists/`

## Key Conventions

1. **Frontmatter required** on all AGENTS, COMMANDS, and SKILLS
2. **Model selection**: `opus` (deep reasoning), `sonnet` (balanced), `haiku` (fast/cheap)
3. **Relative links** in skills: `[references/file.md](references/file.md)`
4. **Skills with scripts**: Use `scripts/run.py` wrapper pattern (see `notebooklm`)

## Skill Categories

| Category | Skills |
|----------|--------|
| **LLM Serving** | vllm, serving-llms-vllm, sglang, tensorrt-llm, llm-serving-patterns, helm-chart-vllm |
| **Inference Optimization** | high-performance-inference, awq, flash-attention, aiperf-benchmark |
| **RAG/Search** | rag-implementation, hybrid-search-implementation, similarity-search-patterns, qdrant, embedding-strategies |
| **MLOps** | implementing-mlops, llm-evaluation, evaluating-llms-harness, langsmith |
| **LLM Development** | langchain-architecture, prompt-engineering-patterns |
| **Kubernetes** | helm-chart-scaffolding, k8s-manifest-generator, k8s-security-policies, operating-kubernetes |
| **Observability** | grafana-dashboards, prometheus-configuration, slo-implementation, analyzing-logs |
| **Python** | async-python-patterns, python-design-patterns, python-error-handling, python-testing-patterns |
| **DevOps** | writing-dockerfiles, implementing-gitops, planning-disaster-recovery |
| **Workflow** | writing-plans, brainstorming, guiding-users, generating-documentation |
| **Debugging** | debug-cuda-crash |
| **Integrations** | notebooklm |

## Agent Overview

| Agent | Model | Purpose |
|-------|-------|---------|
| **Design & Architecture** |||
| docs-architect | opus | Technical documentation generation |
| tdd-orchestrator | opus | TDD workflow orchestration, test-first development |
| **Planning** |||
| kubernetes-architect | opus | K8s/GitOps architecture, EKS/AKS/GKE, service mesh |
| aiops-architect | opus | AI inference infrastructure, model serving orchestration |
| **Development** |||
| ai-engineer | sonnet | LLM applications, RAG systems, agent orchestration |
| fastapi-llm-serving-pro | sonnet | FastAPI LLM serving with vLLM integration |
| vector-database-engineer | sonnet | Vector search, embedding optimization |
| mcp-developer | sonnet | MCP server development |
| bash-pro | sonnet | Shell scripting, automation |
| cpp-pro | sonnet | C++ development, performance optimization |
| rust-engineer | sonnet | Rust development, memory safety |
| **Review & Quality** |||
| code-reviewer | opus | Security vulnerabilities, performance analysis |
| architect-reviewer | opus | System design validation, scalability analysis |
| qa-expert | opus | Testing strategies, quality assurance |
| **Operations** |||
| debugger | sonnet | Root cause analysis, systematic debugging |
| devops-troubleshooter | sonnet | Infrastructure issue diagnosis |
| refactoring-specialist | sonnet | Code improvement, technical debt reduction |
| git-workflow-manager | sonnet | Git operations, branching strategies |
| prompt-engineer | sonnet | Prompt optimization, LLM tuning |

## Usage Patterns

**Invoke agent**: `use <agent-name> to <task>`
**Invoke skill**: `/<skill-name> <task>`
**Invoke command**: `/<command-name> [arguments]`

## Agent-Skill Integration

Several agents come pre-configured with recommended skills for optimal effectiveness:

| Agent | Integrated Skills |
|-------|-------------------|
| fastapi-llm-serving-pro | vllm, serving-llms-vllm, sglang, tensorrt-llm, high-performance-inference, aiperf-benchmark, evaluating-llms-harness |
| docs-architect | generating-documentation, writing-plans, langchain-architecture |
| tdd-orchestrator | python-testing-patterns, writing-plans, python-design-patterns |
| kubernetes-architect | helm-chart-scaffolding, k8s-manifest-generator, k8s-security-policies, implementing-gitops, planning-disaster-recovery |
| aiops-architect | helm-chart-scaffolding, k8s-manifest-generator, helm-chart-vllm, aiperf-benchmark |
| code-reviewer | python-design-patterns, python-testing-patterns, python-error-handling, k8s-security-policies |
| prompt-engineer | prompt-engineering-patterns, langsmith, evaluating-llms-harness, langchain-architecture |

## MCP Server Setup

Configure MCP servers for enhanced Claude Code capabilities:

**Kubernetes:**
```bash
claude mcp add k8s -e KUBECONFIG=~/.kube/config -- npx -y @modelcontextprotocol/server-kubernetes --read-only
```

**Docker:**
```bash
claude mcp add docker -- npx -y @modelcontextprotocol/server-docker
```

## Adding New Content

**New skill**: Create `SKILLS/<name>/SKILL.md` with required frontmatter
**New agent**: Create `AGENTS/<name>.md` with required frontmatter
**New command**: Create `COMMANDS/<name>.md` with required frontmatter

## Workflow Best Practices

### End-to-End LLM Deployment
Recommended Claude Code invocation for complex workflows:
```bash
claude -r "session-name" --model opus --resume --allow-dangerously-skip-permissions
```

**Typical phases:**
1. Research & Ideation → `/notebooklm`, `/brainstorming`
2. Architecture & Design → `docs-architect`, `architect-reviewer`
3. Planning → `/writing-plans`, `/planning-disaster-recovery`
4. Implementation → Use `aiops-architect` or `fastapi-llm-serving-pro` with integrated skills
5. Review & Quality → `code-reviewer`, `refactoring-specialist`, `/tech-debt`
6. Benchmarking & Debugging → `/aiperf-benchmark`, `devops-troubleshooter`, `/debug-cuda-crash`
7. Documentation → `/generating-documentation`, `/commit`
8. Model Evaluation → `fastapi-llm-serving-pro` with `evaluating-llms-harness`

### Git Workflow (from RULES/git-workflow.md)
- **Commit format**: `<type>: <description>` (types: feat, fix, refactor, docs, test, chore, perf, ci)
- **PRs**: Analyze full commit history with `git diff [base-branch]...HEAD`
- **TDD approach**: Write tests first (RED) → Implement (GREEN) → Refactor (IMPROVE)
- **Code review**: Use `code-reviewer` agent immediately after writing code

### Performance Optimization (from RULES/performance.md)
- **Haiku 4.5**: Lightweight agents, frequent invocations (3x cost savings)
- **Sonnet 4.5**: Main development, orchestration (best coding model)
- **Opus 4.5**: Complex architecture, maximum reasoning
- **Context management**: Avoid last 20% of context for large refactors
- **Complex tasks**: Use `ultrathink` + Plan Mode with multiple critique rounds
