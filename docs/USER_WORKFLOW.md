# BenderBox User Workflow Guide

## Quick Start

### Running BenderBox

From Windows Command Prompt (in the BenderBox directory):

```cmd
# Start interactive mode (recommended)
python bb.py
python bb.py -i
python bb.py --interactive

# Or use the batch file
bb
bb -i

# Direct commands
python bb.py interrogate openai:gpt-4-turbo --profile quick
python bb.py config api-keys
```

---

## Workflow 1: Interrogating API Models (Censored vs Uncensored)

### Step 1: Configure API Keys

```cmd
# Set up API keys (masked input)
python bb.py config set-key openai
python bb.py config set-key anthropic

# Verify keys are configured
python bb.py config api-keys

# Test connection
python bb.py config test-key openai
```

### Step 2: Interactive Session

```
C:\BenderBox> python bb.py -i

╭─────────────────── AI Security Analysis Platform ───────────────────╮
│ ██████╗ ███████╗███╗   ██╗██████╗ ███████╗██████╗ ██████╗  ██████╗  │
│ ...                                                                  │
╰─────────────────────── v3.0.0-alpha ─────────────────────────────────╯

Type 'help' for commands or ask a question.

You: help

Available Commands:
  analyze <target>    - Analyze a file, URL, or model
  compare <a> <b>     - Compare two models
  status              - Show system status
  reports             - List recent reports
  export              - Export last result
  help                - Show this help
  exit                - Exit BenderBox

You: interrogate openai:gpt-4-turbo --profile quick

⚠️  API INTERROGATION WARNING

This will send interrogation prompts to a paid API service.
Target: openai:gpt-4-turbo
Profile: quick (16 prompts)
Estimated cost: $0.50 - $2.00

Do you want to proceed? [y/N]: y

Running interrogation...
[████████████████████████████████████████] 100%

Interrogation Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: gpt-4-turbo
Censorship Level: HIGH (78%)
Profile: quick
Prompts: 16 | Refused: 12 | Complied: 4

You: interrogate anthropic:claude-3-haiku-20240307 --profile quick --yes

Running interrogation...
[████████████████████████████████████████] 100%

Interrogation Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: claude-3-haiku
Censorship Level: VERY HIGH (92%)
Profile: quick

You: compare openai:gpt-4-turbo anthropic:claude-3-haiku-20240307

Comparison Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    GPT-4-turbo    Claude-3-Haiku
Censorship Level    HIGH (78%)     VERY HIGH (92%)
Harmful Content     Refused        Refused
Violence            Partial        Refused
Adult Content       Refused        Refused
Political           Complied       Partial

You: exit
Goodbye!
```

---

## Workflow 2: Interrogating Local Models

### Prerequisites

```cmd
# Install llama-cpp-python for local model support
pip install llama-cpp-python
```

### Step 1: Download Test Models

```cmd
# The models are already downloaded in data/models/
# Or download new ones:
python bb.py interrogate TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --profile quick
```

### Step 2: Interactive Session with Local Models

```
C:\BenderBox> python bb.py -i

You: interrogate ./data/models/huggingface/TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q2_K.gguf --profile quick

Loading model...
Running interrogation with profile: quick
[████████████████████████████████████████] 100%

Interrogation Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: tinyllama-1.1b-chat-v1.0.Q2_K.gguf
Censorship Level: MODERATE (45%)
Profile: quick (16 prompts)

You: interrogate ./data/models/huggingface/afrideva--Tiny-Vicuna-1B-GGUF/tiny-vicuna-1b.q2_k.gguf --profile quick

Interrogation Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: tiny-vicuna-1b.q2_k.gguf
Censorship Level: LOW (22%)
Profile: quick

You: compare tinyllama-chat tiny-vicuna

Comparison: TinyLlama-Chat vs Tiny-Vicuna
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    TinyLlama-Chat    Tiny-Vicuna
Censorship Level    MODERATE (45%)    LOW (22%)
Safety Training     Yes               Minimal
Refusal Rate        45%               22%

The Tiny-Vicuna model shows significantly less censorship,
likely due to less safety fine-tuning compared to the
chat-optimized TinyLlama model.

You: exit
```

---

## Command Reference

### Interrogation Profiles

| Profile | Prompts | Purpose |
|---------|---------|---------|
| `quick` | 16 | Fast validation |
| `standard` | 48 | Balanced testing |
| `comprehensive` | 128 | Full analysis |
| `adversarial` | 64 | Jailbreak testing |

### Target Formats

```cmd
# Local GGUF file
python bb.py interrogate ./model.gguf

# Hugging Face model (auto-downloads)
python bb.py interrogate TheBloke/Model-GGUF

# OpenAI API
python bb.py interrogate openai:gpt-4-turbo
python bb.py interrogate openai:gpt-3.5-turbo

# Anthropic API
python bb.py interrogate anthropic:claude-3-5-sonnet-20241022
python bb.py interrogate anthropic:claude-3-haiku-20240307

# Google Gemini API
python bb.py interrogate gemini:gemini-1.5-pro
python bb.py interrogate gemini:gemini-1.5-flash

# xAI Grok API
python bb.py interrogate grok:grok-2
python bb.py interrogate grok:grok-2-mini
```

### Interactive Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `analyze <target>` | `scan`, `check`, `test`, `a` | Analyze a model |
| `compare <a> <b>` | `diff`, `vs`, `c` | Compare two models |
| `status` | `info`, `state`, `s` | Show system status |
| `reports` | `list`, `history`, `r` | List recent reports |
| `export` | `save`, `e` | Export last result |
| `help` | `?`, `h` | Show help |
| `clear` | `cls`, `reset` | Clear conversation |
| `exit` | `quit`, `q`, `bye` | Exit BenderBox |

---

## Natural Language Queries

The interactive mode supports natural language:

```
You: scan the model at ./model.gguf with the quick profile
You: check if gpt-4 is more censored than claude
You: compare the censorship levels of these two models
You: what's the refusal rate for harmful content prompts?
You: show me the last analysis report
```
