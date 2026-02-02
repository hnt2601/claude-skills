# Claude Code Router — Multi-Provider Setup & Model Switching Guide

A complete guide to configuring [claude-code-router](https://github.com/musistudio/claude-code-router) with multiple AI providers (DeepSeek, GLM, Kimi, OpenRouter) and dynamically switching between models.

## What is Claude Code Router?

Claude Code Router (CCR) is a proxy layer that sits between Claude Code CLI and various LLM API providers. It lets you route Claude Code requests to any OpenAI-compatible model — DeepSeek, GLM, Kimi, Gemini, and more — while still using the full Claude Code tooling experience. You can also set up automatic routing rules so different types of tasks (reasoning, long context, background) go to different models.

---

## 1. Installation

### Prerequisites

- Node.js 18+
- Claude Code CLI installed

### Install Claude Code (if not already)

```bash
npm install -g @anthropic-ai/claude-code
```

### Install Claude Code Router

```bash
npm install -g @musistudio/claude-code-router
```

---

## 2. Configuration

All configuration lives in `~/.claude-code-router/config.json`. Create this file manually or use the UI (`ccr ui`).

### Full Example Config

Below is a production-ready config with all four providers — DeepSeek, GLM, Kimi (via SiliconFlow), and OpenRouter:

```json
{
  "LOG": true,
  "API_TIMEOUT_MS": 600000,
  "Providers": [
    {
      "name": "deepseek",
      "api_base_url": "https://api.deepseek.com/chat/completions",
      "api_key": "$DEEPSEEK_API_KEY",
      "models": ["deepseek-chat", "deepseek-reasoner"],
      "transformer": {
        "use": ["deepseek"],
        "deepseek-chat": {
          "use": ["tooluse"]
        }
      }
    },
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "$OPENROUTER_API_KEY",
      "models": [
        "google/gemini-2.5-pro-preview",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-chat-v3-0324:free"
      ],
      "transformer": {
        "use": ["openrouter"]
      }
    },
    {
      "name": "glm",
      "api_base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
      "api_key": "$GLM_API_KEY",
      "models": ["glm-4-plus", "glm-4-flash"]
    },
    {
      "name": "kimi",
      "api_base_url": "https://api.siliconflow.cn/v1/chat/completions",
      "api_key": "$SILICONFLOW_API_KEY",
      "models": ["moonshotai/Kimi-K2-Instruct"],
      "transformer": {
        "use": [
          ["maxtoken", { "max_tokens": 16384 }]
        ]
      }
    }
  ],
  "Router": {
    "default": "deepseek,deepseek-chat",
    "think": "deepseek,deepseek-reasoner",
    "longContext": "openrouter,google/gemini-2.5-pro-preview",
    "longContextThreshold": 60000
  }
}
```

---

## 3. Provider Configuration Details

### DeepSeek

| Field | Value |
|-------|-------|
| API Base URL | `https://api.deepseek.com/chat/completions` |
| Models | `deepseek-chat` (general), `deepseek-reasoner` (reasoning/R1) |
| Transformer | `deepseek` (required) |
| Get API Key | [platform.deepseek.com](https://platform.deepseek.com/) |

The `deepseek` transformer handles format conversion between Claude Code's Anthropic-style requests and DeepSeek's API. For `deepseek-chat`, the additional `tooluse` transformer optimizes tool calling via `tool_choice`.

```json
{
  "name": "deepseek",
  "api_base_url": "https://api.deepseek.com/chat/completions",
  "api_key": "sk-your-key",
  "models": ["deepseek-chat", "deepseek-reasoner"],
  "transformer": {
    "use": ["deepseek"],
    "deepseek-chat": {
      "use": ["tooluse"]
    }
  }
}
```

### OpenRouter

| Field | Value |
|-------|-------|
| API Base URL | `https://openrouter.ai/api/v1/chat/completions` |
| Models | Any model on OpenRouter (Gemini, Claude, Llama, Mistral, etc.) |
| Transformer | `openrouter` (required) |
| Get API Key | [openrouter.ai/keys](https://openrouter.ai/keys) |

OpenRouter acts as a unified gateway to 200+ models. One API key gives you access to models from Google, Anthropic, Meta, DeepSeek, and more. The `openrouter` transformer is required for all models accessed through OpenRouter.

```json
{
  "name": "openrouter",
  "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
  "api_key": "sk-or-your-key",
  "models": [
    "google/gemini-2.5-pro-preview",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.5-sonnet"
  ],
  "transformer": {
    "use": ["openrouter"]
  }
}
```

You can also configure provider routing preferences for specific models:

```json
"transformer": {
  "use": ["openrouter"],
  "moonshotai/kimi-k2": {
    "use": [
      ["openrouter", {
        "provider": {
          "only": ["moonshotai/fp8"]
        }
      }]
    ]
  }
}
```

### GLM (Zhipu AI)

| Field | Value |
|-------|-------|
| API Base URL | `https://open.bigmodel.cn/api/paas/v4/chat/completions` |
| Models | `glm-4-plus`, `glm-4-flash`, etc. |
| Transformer | None required (OpenAI-compatible) |
| Get API Key | [open.bigmodel.cn](https://open.bigmodel.cn/) |

GLM-4 series uses an OpenAI-compatible API format, so no special transformer is needed for basic usage.

```json
{
  "name": "glm",
  "api_base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
  "api_key": "your-glm-key",
  "models": ["glm-4-plus", "glm-4-flash"]
}
```

**Alternative: GLM via AIHubMix**

If you prefer using AIHubMix as a proxy:

```json
{
  "name": "aihubmix",
  "api_base_url": "https://aihubmix.com/v1/chat/completions",
  "api_key": "sk-aihubmix-key",
  "models": ["Z/glm-4.5"]
}
```

**Alternative: GLM Coding Plan via Z.ai**

Z.ai offers a GLM Coding Plan starting at $3/month with access to GLM-4.6:

```json
{
  "name": "z-glm",
  "api_base_url": "https://open.z.ai/api/paas/v4/chat/completions",
  "api_key": "your-z-api-key",
  "models": ["glm-4.6"]
}
```

### Kimi (Moonshot AI via SiliconFlow)

| Field | Value |
|-------|-------|
| API Base URL | `https://api.siliconflow.cn/v1/chat/completions` |
| Models | `moonshotai/Kimi-K2-Instruct` |
| Transformer | `maxtoken` (recommended to cap output) |
| Get API Key | [cloud.siliconflow.cn](https://cloud.siliconflow.cn/) |

Kimi-K2 is hosted on SiliconFlow. The `maxtoken` transformer caps the output token limit to prevent excessive generation.

```json
{
  "name": "kimi",
  "api_base_url": "https://api.siliconflow.cn/v1/chat/completions",
  "api_key": "sk-siliconflow-key",
  "models": ["moonshotai/Kimi-K2-Instruct"],
  "transformer": {
    "use": [
      ["maxtoken", { "max_tokens": 16384 }]
    ]
  }
}
```

---

## 4. Router Configuration (Automatic Routing)

The `Router` section determines which model handles which type of task automatically:

```json
"Router": {
  "default": "deepseek,deepseek-chat",
  "think": "deepseek,deepseek-reasoner",
  "background": "glm,glm-4-flash",
  "longContext": "openrouter,google/gemini-2.5-pro-preview",
  "longContextThreshold": 60000,
  "webSearch": "openrouter,google/gemini-2.5-flash:online"
}
```

| Router Key | Purpose | When It Triggers |
|------------|---------|------------------|
| `default` | General-purpose tasks | All requests unless another rule matches |
| `think` | Reasoning-heavy tasks | Plan Mode, complex analysis |
| `background` | Background/lightweight tasks | Automated sub-tasks |
| `longContext` | Long context handling | When token count exceeds `longContextThreshold` |
| `longContextThreshold` | Token threshold for long context | Default: 60000 tokens |
| `webSearch` | Web search tasks | When web search is invoked |
| `image` | Image-related tasks (beta) | Image processing via built-in agent |

The format is always `provider_name,model_name`.

---

## 5. Running Claude Code with the Router

### Option A: All-in-one (recommended)

```bash
ccr code
```

This starts the router proxy and launches Claude Code in one command.

### Option B: Separate processes

```bash
# Start the router in the background
ccr start

# Set environment variables for the current shell
eval "$(ccr activate)"

# Now use claude normally — requests are routed through CCR
claude
```

To make the environment variables persistent, add to your shell profile:

```bash
# Add to ~/.zshrc or ~/.bashrc
eval "$(ccr activate)"
```

---

## 6. Switching Models

### Method 1: `/model` command (inside Claude Code)

While in a Claude Code session, type:

```
/model deepseek,deepseek-chat
/model deepseek,deepseek-reasoner
/model openrouter,google/gemini-2.5-pro-preview
/model openrouter,anthropic/claude-sonnet-4
/model glm,glm-4-plus
/model kimi,moonshotai/Kimi-K2-Instruct
```

The format is `/model <provider_name>,<model_name>`. The switch takes effect immediately for subsequent requests.

### Method 2: Interactive CLI

```bash
ccr model
```

This opens an interactive terminal UI where you can:

- View current model assignments for each router type
- Switch models for default, think, background, longContext, etc.
- Add new models to existing providers
- Create entirely new provider configurations

### Method 3: Web UI

```bash
ccr ui
```

Opens a browser-based interface for visual config editing. No JSON editing required.

---

## 7. Environment Variable Interpolation

For security, avoid hardcoding API keys in config. Use environment variable references:

```json
{
  "Providers": [
    {
      "name": "deepseek",
      "api_key": "$DEEPSEEK_API_KEY"
    },
    {
      "name": "openrouter",
      "api_key": "${OPENROUTER_API_KEY}"
    }
  ]
}
```

Then export the variables in your shell:

```bash
export DEEPSEEK_API_KEY="sk-xxx"
export OPENROUTER_API_KEY="sk-or-xxx"
export GLM_API_KEY="your-glm-key"
export SILICONFLOW_API_KEY="sk-sf-xxx"
```

Both `$VAR_NAME` and `${VAR_NAME}` syntax are supported. Interpolation works recursively through nested objects and arrays.

---

## 8. Available Transformers

| Transformer | Purpose |
|-------------|---------|
| `deepseek` | Adapts requests/responses for DeepSeek API |
| `openrouter` | Adapts requests/responses for OpenRouter API |
| `gemini` | Adapts requests/responses for Google Gemini API |
| `groq` | Adapts requests/responses for Groq API |
| `Anthropic` | Preserves original Anthropic format (pass-through) |
| `tooluse` | Optimizes tool calling via `tool_choice` |
| `maxtoken` | Sets a specific `max_tokens` limit |
| `reasoning` | Processes the `reasoning_content` field |
| `sampling` | Processes sampling params (temperature, top_p, etc.) |
| `enhancetool` | Adds error tolerance to tool call params (disables streaming) |
| `cleancache` | Removes `cache_control` field from requests |
| `vertex-gemini` | Handles Gemini API with Vertex authentication |

Custom transformers can be loaded via the `transformers` field in config:

```json
{
  "transformers": [
    {
      "path": "/path/to/custom-transformer.js",
      "options": { "key": "value" }
    }
  ]
}
```

---

## 9. Useful Commands Reference

| Command | Description |
|---------|-------------|
| `ccr code` | Start router + launch Claude Code |
| `ccr start` | Start router as background service |
| `ccr restart` | Restart router (required after config changes) |
| `ccr model` | Interactive model selector |
| `ccr ui` | Open web-based config editor |
| `eval "$(ccr activate)"` | Set env vars for current shell |

---

## 10. Recommended Routing Strategy

For a cost-effective and performant setup:

| Task Type | Recommended Provider/Model | Rationale |
|-----------|---------------------------|-----------|
| Default (general coding) | DeepSeek Chat | Fast, cheap, good code quality |
| Reasoning/Planning | DeepSeek Reasoner | Strong reasoning at low cost |
| Long Context (>60K tokens) | Gemini 2.5 Pro (via OpenRouter) | 1M+ token context window |
| Background tasks | GLM-4 Flash | Lightweight and fast |
| Web Search | Gemini 2.5 Flash (via OpenRouter) | Native web search support |

This setup minimizes API costs while ensuring the best model is used for each task type.

---

## 11. Troubleshooting

**Config changes not taking effect?** Always restart after editing config:

```bash
ccr restart
```

**Connection errors?** Check that the router is running:

```bash
# The router runs on localhost:3456 by default
curl http://127.0.0.1:3456/health
```

**API key issues?** Verify environment variables are exported:

```bash
echo $DEEPSEEK_API_KEY
```

**Check logs for detailed errors:**

```bash
# Server logs
cat ~/.claude-code-router/logs/ccr-*.log

# Application logs
cat ~/.claude-code-router/claude-code-router.log
```

**Proxy needed?** Add proxy config:

```json
{
  "PROXY_URL": "http://127.0.0.1:7890"
}
```

---

## References

- [claude-code-router GitHub](https://github.com/musistudio/claude-code-router)
- [DeepSeek API Docs](https://platform.deepseek.com/api-docs)
- [OpenRouter Docs](https://openrouter.ai/docs)
- [Zhipu AI / GLM Docs](https://open.bigmodel.cn/dev/howuse/introduction)
- [SiliconFlow Docs](https://docs.siliconflow.cn/)