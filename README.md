# AI Coding Agent — VS Code Extension

An agentic AI coding assistant for VS Code. 

**Why I built this:** I created this project for my personal use because premium AI coding tools (like Cursor, GitHub Copilot) are expensive, and I wanted a fully functional, free alternative. 

This agent uses **Chain of Thought (CoT)** step-by-step reasoning and a robust fallback system utilizing high-limit, free cloud LLM providers. Currently powered by **Cerebras Llama 3.1 70B/8B** (primary for blazing fast inference), with automatic fallbacks to **Gemini 2.0 Flash**, **Groq**, and **OpenRouter** to ensure it never gets rate-limited.
## Architecture

```
VS Code Extension (TypeScript) ←→ Flask Server (Python) ←→ Gemini / Groq API
```

- **Extension**: Sidebar webview chat UI, file writing, terminal commands
- **Server**: Agentic tool-calling loop with 8 tools (list files, read, write, search, run commands, SQLite, etc.)

## Setup

### 1. Install Python Dependencies

```bash
cd ai-coder
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `.env` with your keys:

```env
GEMINI_API_KEY=your_gemini_key    # Free at https://aistudio.google.com/apikey
GROQ_API_KEY=your_groq_key       # Free at https://console.groq.com
```

### 3. Start the Server

```bash
python server.py
```

### 4. Install & Run the Extension

```bash
cd extension
npm install
npm run compile
```

Then press **F5** in VS Code to launch the Extension Development Host.

## Tools

| Tool | Permission | Description |
|------|-----------|-------------|
| `list_files` | Auto | List project files |
| `read_file` | Auto | Read file contents |
| `search_files` | Auto | Grep-like search |
| `read_database` | Auto | SQLite schema + sample |
| `run_db_query` | Auto | Run SQL query |
| `write_file` | ⚠️ Ask | Create/edit files |
| `run_command` | ⚠️ Ask | Terminal commands |
| `ask_user` | ⚠️ Ask | Clarifying questions |
