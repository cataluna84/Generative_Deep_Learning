# Walkthrough: AGENTS.md Configuration Task

**Date**: December 19, 2025  
**Duration**: ~30 minutes

---

## Objective

Create a comprehensive `agents.md` file for the Generative Deep Learning project that serves as a machine-readable briefing for AI coding agents.

---

## Research Phase

### Tools Used
- **Exa web search** - Found agents.md specification and examples
- **Exa crawling** - Extracted content from Factory AI docs and agents.md website
- **Ref documentation search** - Found Gemini CLI configuration docs

### Key Sources Consulted
1. [Factory AI agents.md specification](https://docs.factory.ai/cli/configuration/agents-md)
2. [agents.md standard](https://agents.md/) - Used by 60k+ open-source projects
3. [GitHub Blog: Lessons from 2,500 repos](https://github.blog/ai-and-ml/github-copilot/how-to-write-a-great-agents-md-lessons-from-over-2500-repositories/)
4. [Google Gemini CLI docs](https://google-gemini.github.io/gemini-cli/docs/cli/gemini-md.html)

### Findings
- AGENTS.md is a cross-platform standard (works with Cursor, Aider, Gemini CLI, Codex, Jules, etc.)
- Best practice: Put setup commands early, define clear boundaries
- Gemini CLI uses `GEMINI.md` by default but supports AGENTS.md via `settings.json`

---

## Implementation Phase

### Files Created

| File | Purpose |
|------|---------|
| `agents.md` | Main agent briefing file at project root |
| `documentation/implementation_plan.md` | Research and planning documentation |
| `documentation/walkthrough.md` | This file - task summary |

### AGENTS.md Structure

1. **Setup Commands** - `uv sync`, Python verification (placed first per best practices)
2. **Development Commands** - Jupyter Lab, script execution, data downloads
3. **Project Architecture** - Directory structure with responsibilities
4. **Code Conventions** - Naming, imports, Keras 3.0+ patterns
5. **Domain Vocabulary** - ML/DL terminology table
6. **Gotchas & Boundaries** - Do NOT / Always sections
7. **Gemini CLI Configuration** - settings.json snippet

---

## Verification

All files created successfully:
- ✅ `agents.md` at project root
- ✅ `documentation/implementation_plan.md`
- ✅ `documentation/walkthrough.md`

---

## Next Steps

1. Optionally configure Gemini CLI by adding `.gemini/settings.json`
2. Update AGENTS.md as project evolves
3. Consider adding subdirectory-specific AGENTS.md for complex modules
