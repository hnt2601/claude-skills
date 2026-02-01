# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Claude Code skills library** - a collection of reusable agents, commands, rules, and domain-specific skills for Claude Code. The repository contains no executable code; it's a reference library of markdown-based configurations and prompts.

## Directory Structure

```
├── AGENTS/      # Reusable agent definitions with model and description frontmatter
├── COMMANDS/    # Slash command templates with $ARGUMENTS placeholder
├── RULES/       # Behavioral guidelines (git-workflow, performance optimization)
├── SKILLS/      # Domain-specific knowledge bases (30 skills)
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
allowed-tools: Bash(git add:*), Bash(git commit:*)  # Optional: restrict tool access
description: "Command description"
argument-hint: "<required-arg> [optional-arg]"  # Optional
---
```
Commands may include:
- `$ARGUMENTS` placeholder for user-provided arguments
- `!`backtick commands for dynamic context injection (e.g., `!`git status`)

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

- **ML/AI Operations**: implementing-mlops, llm-evaluation, llm-serving-patterns, embedding-strategies, evaluating-llms-harness
- **Inference Optimization**: high-performance-inference, vllm, serving-llms-vllm, sglang, tensorrt-llm, awq, flash-attention
- **RAG/Search**: rag-implementation, hybrid-search-implementation, similarity-search-patterns, qdrant
- **Kubernetes**: helm-chart-scaffolding, k8s-manifest-generator, k8s-security-policies
- **Python Patterns**: async-python-patterns, python-design-patterns, python-error-handling, python-testing-patterns
- **LLM Development**: langsmith, langchain-architecture, prompt-engineering-patterns, debug-cuda-crash
- **Workflow/Planning**: writing-plans, brainstorming
- **Integrations**: notebooklm (browser automation for Google NotebookLM queries)

## Available Agents

Agents in `AGENTS/` are specialized for different tasks:
- **code-reviewer**: Code quality, security, performance analysis (opus)
- **docs-architect**: Documentation generation
- **kubernetes-architect**: K8s infrastructure design
- **architect-reviewer**: Architecture review
- **debugger**: Debugging assistance
- **git-workflow-manager**: Git operations and workflows
- **prompt-engineer**: Prompt optimization
- **qa-expert**: Testing and quality assurance
- **refactoring-specialist**: Code refactoring
- **mcp-developer**: MCP server development
- **cpp-pro**, **rust-engineer**: Language-specific expertise

## Workflow Guidelines (from RULES/)

RULES/ contains behavioral guidelines that may include CLAUDE.md files for auto-context loading.

### Git Workflow
- Commit format: `<type>: <description>` (types: feat, fix, refactor, docs, test, chore, perf, ci)
- TDD approach: write tests first (RED) → implement (GREEN) → refactor (IMPROVE)
- Use code-reviewer agent after writing code

### Performance Optimization
- Use Haiku for lightweight/frequent agent invocations
- Use Sonnet for main development work
- Use Opus for complex architectural decisions
- Avoid last 20% of context window for large refactoring tasks

## Skills with Executable Scripts

Some skills contain runnable automation scripts:
- **notebooklm**: Python scripts for Google NotebookLM browser automation (`scripts/run.py` wrapper pattern)
