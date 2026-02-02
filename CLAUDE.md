# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Claude Code skills library** - a collection of reusable agents, commands, rules, and domain-specific skills for Claude Code. The repository contains no executable code; it's a reference library of markdown-based configurations and prompts.

## Directory Structure

```
├── AGENTS/      # Agent definitions (17 agents)
├── COMMANDS/    # Slash command templates (5 commands)
├── RULES/       # Behavioral guidelines
├── SKILLS/      # Domain-specific knowledge bases (42 skills)
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
| code-reviewer | opus | Security, performance, code quality |
| architect-reviewer | opus | System design validation |
| docs-architect | opus | Technical documentation |
| tdd-orchestrator | opus | TDD workflow orchestration |
| kubernetes-architect | opus | K8s/GitOps architecture |
| qa-expert | opus | Testing strategies |
| ai-engineer | sonnet | LLM applications, RAG systems |
| vector-database-engineer | sonnet | Vector search implementation |
| devops-troubleshooter | sonnet | Infrastructure debugging |
| debugger | sonnet | Root cause analysis |
| refactoring-specialist | sonnet | Code improvement |
| git-workflow-manager | sonnet | Git operations |
| prompt-engineer | sonnet | Prompt optimization |
| mcp-developer | sonnet | MCP server development |
| bash-pro | sonnet | Shell scripting |
| cpp-pro | sonnet | C++ development |
| rust-engineer | sonnet | Rust development |

## Usage Patterns

**Invoke agent**: `use <agent-name> to <task>`
**Invoke skill**: `/<skill-name> <task>`
**Invoke command**: `/<command-name> [arguments]`

## Adding New Content

**New skill**: Create `SKILLS/<name>/SKILL.md` with required frontmatter
**New agent**: Create `AGENTS/<name>.md` with required frontmatter
**New command**: Create `COMMANDS/<name>.md` with required frontmatter
