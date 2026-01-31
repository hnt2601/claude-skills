# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Claude Code skills library** - a collection of reusable agents, commands, rules, and domain-specific skills for Claude Code. The repository contains no executable code; it's a reference library of markdown-based configurations and prompts.

## Directory Structure

```
├── AGENTS/      # Reusable agent definitions with model and description frontmatter
├── COMMANDS/    # Slash command templates with $ARGUMENTS placeholder
├── RULES/       # Behavioral guidelines (git-workflow, performance optimization)
├── SKILLS/      # Domain-specific knowledge bases (31 skills)
```

## File Formats

### AGENTS (`AGENTS/*.md`)
Agent definitions with YAML frontmatter:
```yaml
---
name: agent-name
description: Brief description for Claude Code to select appropriate agent
model: opus | sonnet | haiku
---
```
Body contains: purpose, capabilities, behavioral traits, knowledge base, and response approach.

### COMMANDS (`COMMANDS/*.md`)
Slash command templates with frontmatter:
```yaml
---
description: "Command description"
argument-hint: "<required-arg> [optional-arg]"
---
```
Use `$ARGUMENTS` placeholder in the body for user-provided arguments.

### SKILLS (`SKILLS/<skill-name>/`)
Each skill is a self-contained knowledge base:
- `SKILL.md` - Main skill definition with YAML frontmatter (name, description)
- `references/` - Deep-dive documentation files
- `examples/` - Code examples (Python, YAML, etc.)
- `assets/` - Templates and static resources
- `scripts/` - Helper scripts
- `checklists/` - Step-by-step guides

Skill YAML frontmatter:
```yaml
---
name: skill-name
description: Detailed description of when to use this skill
---
```

## Key Conventions

1. **Frontmatter is required** - All AGENTS, COMMANDS, and SKILLS use YAML frontmatter for metadata
2. **Model selection** - AGENTS specify model preference: `opus` (deep reasoning), `sonnet` (balanced), `haiku` (fast/cheap)
3. **References are linked** - Skills use relative paths like `[references/feature-stores.md](references/feature-stores.md)`
4. **Decision frameworks** - Skills include structured decision matrices for tool/platform selection

## Current Skills Categories

- **ML/AI Operations**: implementing-mlops, llm-evaluation, llm-serving-patterns, embedding-strategies
- **Inference Optimization**: high-performance-inference, vllm, sglang, tensorrt-llm, awq, flash-attention
- **RAG/Search**: rag-implementation, hybrid-search-implementation, similarity-search-patterns
- **Kubernetes**: helm-chart-scaffolding, k8s-manifest-generator, k8s-security-policies
- **Python Patterns**: async-python-patterns, python-design-patterns, python-error-handling, python-testing-patterns
- **LLM Development**: langsmith, prompt-engineering-patterns, debug-cuda-crash

## Workflow Guidelines (from RULES/)

### Git Workflow
- Commit format: `<type>: <description>` (types: feat, fix, refactor, docs, test, chore, perf, ci)
- TDD approach: write tests first (RED) → implement (GREEN) → refactor (IMPROVE)
- Use code-reviewer agent after writing code

### Performance Optimization
- Use Haiku for lightweight/frequent agent invocations
- Use Sonnet for main development work
- Use Opus for complex architectural decisions
- Avoid last 20% of context window for large refactoring tasks
