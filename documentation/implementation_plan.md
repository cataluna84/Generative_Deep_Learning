# Implementation Plan: AGENTS.md Configuration File

## Goal

Create a comprehensive `agents.md` file for the Generative Deep Learning project that follows:
- [Factory AI specification](https://docs.factory.ai/cli/configuration/agents-md)
- [agents.md standard](https://agents.md/) (used by 60k+ open-source projects)
- [GitHub best practices](https://github.blog/ai-and-ml/github-copilot/how-to-write-a-great-agents-md-lessons-from-over-2500-repositories/) from analysis of 2,500 repositories

---

## Research Summary

### What is AGENTS.md?
- A Markdown "briefing packet" for AI coding agents—separate from README (for humans)
- Works across: **Cursor**, **Aider**, **Gemini CLI**, **OpenAI Codex**, **Jules**, **Factory Droids**, **Zed**

### Google Gemini CLI / Antigravity Compatibility

Gemini CLI uses `GEMINI.md` by default, but supports AGENTS.md via configuration.

**To enable AGENTS.md in Gemini CLI**, add to `.gemini/settings.json`:
```json
{"context":{"fileName":["AGENTS.md","GEMINI.md"]}}
```

**Gemini CLI Context Hierarchy** (all files are concatenated):
1. Global: `~/.gemini/GEMINI.md` (or AGENTS.md)
2. Project root + ancestor directories (up to `.git`)
3. Subdirectories (respects `.gitignore`)

### Key Lessons from GitHub's 2,500 Repo Analysis
| Best Practice | Description |
|---------------|-------------|
| **Be a specialist** | Agents work better with specific personas, not vague helpers |
| **Commands first** | Put exact build/test commands early in the file |
| **Clear boundaries** | Define what the agent should NOT do |
| **Actionable gotchas** | List domain-specific pitfalls with solutions |
| **Testing instructions** | Agents execute commands automatically if listed |

---

## Project Analysis

### Structure
```
Generative_Deep_Learning/
├── models/           # 8 models: AE, VAE, GAN, WGAN, WGANGP, CycleGAN, MuseGAN, RNNAttention
├── utils/            # loaders, callbacks, write utilities
├── scripts/          # Data download scripts
├── data/             # Dataset storage
├── run/              # Model outputs
├── *.ipynb           # 22 notebooks (Ch 02-09)
└── pyproject.toml    # TensorFlow 2.16+, Keras 3.0+, Python 3.13+
```

### Key Dependencies
- **Python** >= 3.13 | **TensorFlow** >= 2.16.0 | **Keras** >= 3.0.0
- **Data Science**: NumPy, Pandas, SciPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Music**: music21 (Chapter 7)

---

## Changes Made

### [CREATED] agents.md
Created comprehensive agents.md file at project root with:
- Setup commands (placed early per best practices)
- Development commands
- Project architecture with directory structure
- Code conventions (Keras 3.0+ patterns)
- Domain vocabulary table
- Gotchas & boundaries (Do NOT / Always sections)
- Gemini CLI configuration instructions

### [CREATED] documentation/implementation_plan.md
This file - documenting the research and implementation

### [CREATED] documentation/walkthrough.md
Summary of what was accomplished

---

## Verification

```powershell
# Verify files exist
Test-Path "c:\Antigravity_Workspace\Generative_Deep_Learning\agents.md"
Test-Path "c:\Antigravity_Workspace\Generative_Deep_Learning\documentation\implementation_plan.md"
Test-Path "c:\Antigravity_Workspace\Generative_Deep_Learning\documentation\walkthrough.md"
```
