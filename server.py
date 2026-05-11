"""
AI Coding Agent — Flask Backend Server
Provides an agentic AI coding assistant with tool calling.
Primary LLM: Gemini 2.0 Flash (free) | Fallback: Groq Llama 3.3 70B
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import os
import json
import sqlite3
import time
import fnmatch
import subprocess
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ─── LLM Provider Configuration ──────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def create_providers():
    """Build list of available LLM providers in fallback order."""
    providers = []

    # 1. Gemini 2.0 Flash — fastest, free tier
    if GEMINI_API_KEY:
        gemini_client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        providers.append({
            "name": "Gemini 2.0 Flash",
            "client": gemini_client,
            "model": "gemini-2.0-flash",
            "max_tokens": 4096
        })
        providers.append({
            "name": "Gemini 2.5 Flash",
            "client": gemini_client,
            "model": "gemini-2.5-flash",
            "max_tokens": 4096
        })

    # 2. Groq — ultra-fast inference, free tier
    if GROQ_API_KEY:
        groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        providers.append({
            "name": "Groq Llama 3.3 70B",
            "client": groq_client,
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 4096
        })
        providers.append({
            "name": "Groq DeepSeek R1 70B",
            "client": groq_client,
            "model": "deepseek-r1-distill-llama-70b",
            "max_tokens": 4096
        })

    # 3. OpenRouter — access to 50+ models, great free-tier options
    if OPENROUTER_API_KEY:
        or_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/ai-coder",
                "X-Title": "AI Coding Agent"
            }
        )
        providers.append({
            "name": "OpenRouter Llama 3.3",
            "client": or_client,
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "max_tokens": 8000
        })
        providers.append({
            "name": "OpenRouter Qwen3",
            "client": or_client,
            "model": "qwen/qwen3-next-80b-a3b-instruct:free",
            "max_tokens": 8000
        })

    # Override providers list with the ones we know work (replacing decommissioned ones)
    final_providers = []
    
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
    if CEREBRAS_API_KEY:
        try:
            from cerebras.cloud.sdk import Cerebras
            cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
            final_providers.append({"name": "Cerebras Llama 3.1 70B", "model": "llama3.1-70b", "client": cerebras_client, "max_tokens": 8192})
            final_providers.append({"name": "Cerebras Llama 3.1 8B", "model": "llama3.1-8b", "client": cerebras_client, "max_tokens": 8192})
        except ImportError:
            print("  ⚠️  Cerebras SDK not installed. Run `pip install cerebras-cloud-sdk`")

    final_providers.extend([
        {"name": "Gemini 2.0 Flash", "model": "gemini-2.0-flash", "client": OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv("GEMINI_API_KEY")), "max_tokens": 8000},
        {"name": "Gemini 2.5 Flash", "model": "gemini-2.5-flash", "client": OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv("GEMINI_API_KEY")), "max_tokens": 8000},
        {"name": "Groq Llama 3.3 70B", "model": "llama-3.3-70b-versatile", "client": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")), "max_tokens": 8000},
        {"name": "Groq Llama 3.1 8B", "model": "llama-3.1-8b-instant", "client": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")), "max_tokens": 8000},
        {"name": "OpenRouter Llama 3.3", "model": "meta-llama/llama-3.3-70b-instruct:free", "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")), "max_tokens": 8000},
        {"name": "OpenRouter Qwen3", "model": "qwen/qwen3-next-80b-a3b-instruct:free", "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")), "max_tokens": 8000},
    ])

    return final_providers


PROVIDERS = create_providers()

# ─── State ────────────────────────────────────────────────────────────────────

conversation_history = {}
stop_flags = {}

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI coding agent running inside VS Code, similar to Cursor or GitHub Copilot. You help users build, fix, and understand code projects.

## Chain of Thought (CoT)
Before you call any tools or output any code, you MUST think step-by-step.
Always enclose your reasoning within a `<thought>` block.
Analyze the user's request, examine the provided [CONTEXT], formulate a plan, and decide which tools to use.
Example:
<thought>
The user wants to update the button color.
I see the active file is `button.tsx`. I should read it first using `read_file`.
</thought>
I will read `button.tsx` to see how the color is currently defined.

## Your Workflow
1. **Understand** — Read the user's request carefully. Check the [CONTEXT] block — it tells you what file the user has open and what they have selected. Use this context first before calling list_files.
2. **Research** — Use list_files to understand the full project structure. Then read_file on any files you need to understand. Use search_files to find specific patterns. Never assume file contents.
3. **Plan** — Explain what you'll do. Be specific about which files you'll change and why.
4. **Execute** — Make changes one file at a time using write_file. Each write requires user approval.
5. **Verify** — After writing files, run_command to test. Read the output. If it fails, self-correct.

## Critical Rules
- If the user just greets you (e.g., "Hi", "Hello") or asks a general non-coding question, respond politely and conversationally. Do NOT call any tools.
- ALWAYS start each response with a COMPLETE standalone sentence. Never continue mid-sentence from a previous message.
- When the user has a [CONTEXT] block, use it — if they ask to "fix this", they mean the active file shown in context.
- Always read a file before modifying it — never guess at contents.
- Write the COMPLETE file — never use placeholders like '// rest of code here'.
- Use **bold**, `backticks`, and fenced code blocks in your explanations.
- For multi-file changes, handle one file at a time.

## Self-Correction Loop
- If run_command returns an error, analyze it and fix the code. Re-run up to 3 times before asking the user.
- Never give up after a single failure — errors are expected and fixable.
- If you write a file and the command fails with a syntax error, read the file back and fix it.

## Tool Usage Tips
- Use `search_files` to find where a function/class is defined instead of reading every file.
- Use `read_file_lines` to read just a specific section of a large file instead of the whole thing.
- Use `ask_user` only when you genuinely cannot proceed without human input."""

# ─── Tool Definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the workspace to understand the project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "description": "Relative folder path. Use '.' for root."}
                },
                "required": ["folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace root."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file. Requires user permission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace root."},
                    "content": {"type": "string", "description": "Complete file content to write."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a terminal/shell command. Requires user permission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run."},
                    "reason": {"type": "string", "description": "Why this command is needed."}
                },
                "required": ["command", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for text across files in the project. Like grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for (case-insensitive)."},
                    "file_pattern": {"type": "string", "description": "Optional glob like '*.py' or '*.js'."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_database",
            "description": "Read the schema and sample rows from a SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to .db file relative to workspace."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_db_query",
            "description": "Run a SQL query on a SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to .db file."},
                    "query": {"type": "string", "description": "SQL query to execute."}
                },
                "required": ["path", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a clarifying question before proceeding. Only use when you genuinely cannot proceed without their input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask."}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_lines",
            "description": "Read a specific line range from a file. Use instead of read_file for large files when you only need a section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace root."},
                    "start_line": {"type": "integer", "description": "Starting line number (1-based)."},
                    "end_line": {"type": "integer", "description": "Ending line number (1-based, inclusive)."}
                },
                "required": ["path", "start_line", "end_line"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Perform semantic/vector search across the codebase to find relevant code snippets using RAG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query (e.g., 'where is the user authentication logic?')"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "index_workspace",
            "description": "Initialize or update the Vector DB (RAG) index for the current workspace. Call this before using search_code.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Run 'git status' to see changed, untracked, and staged files.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Run 'git diff' to see uncommitted changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Optional specific file to diff."}
                },
                "required": []
            }
        }
    }
]

# ─── Tool Execution ──────────────────────────────────────────────────────────

SKIP_DIRS = {'.git', 'venv', 'env', '.venv', 'node_modules', '__pycache__', 'dist', 'build', '.next', '.cache'}


def execute_tool(tool_name, tool_args, workspace):
    """Execute a tool and return the result dict."""

    if tool_name == "list_files":
        folder = tool_args.get("folder", ".")
        full_path = os.path.join(workspace, folder)
        try:
            files = []
            for root, dirs, filenames in os.walk(full_path):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
                for f in filenames:
                    if f.startswith('.'):
                        continue
                    rel = os.path.relpath(os.path.join(root, f), workspace)
                    files.append(rel.replace('\\', '/'))
            return {"status": "success", "files": sorted(files)[:100]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif tool_name == "read_file":
        filepath = os.path.join(workspace, tool_args["path"])
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.splitlines()
            total_lines = len(lines)
            if len(content) > 8000:
                content = content[:8000] + f"\n\n...(truncated — {total_lines} lines total, use read_file_lines to read specific sections)"
            return {"status": "success", "content": content, "path": tool_args["path"], "total_lines": total_lines}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif tool_name == "read_file_lines":
        filepath = os.path.join(workspace, tool_args["path"])
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            start = max(0, tool_args["start_line"] - 1)
            end = min(len(lines), tool_args["end_line"])
            content = "".join(lines[start:end])
            return {
                "status": "success",
                "content": content,
                "path": tool_args["path"],
                "start_line": start + 1,
                "end_line": end,
                "total_lines": len(lines)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif tool_name == "search_files":
        query = tool_args.get("query", "")
        pattern = tool_args.get("file_pattern", None)
        matches = []
        try:
            for root, dirs, filenames in os.walk(workspace):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
                for f in filenames:
                    if pattern and not fnmatch.fnmatch(f, pattern):
                        continue
                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                            for line_num, line in enumerate(fh, 1):
                                if query.lower() in line.lower():
                                    rel = os.path.relpath(filepath, workspace).replace('\\', '/')
                                    matches.append({"file": rel, "line": line_num, "text": line.strip()[:200]})
                                    if len(matches) >= 30:
                                        return {"status": "success", "matches": matches, "query": query}
                    except:
                        pass
            return {"status": "success", "matches": matches, "query": query}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif tool_name == "read_database":
        filepath = os.path.join(workspace, tool_args["path"])
        try:
            conn = sqlite3.connect(filepath)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            schema_info = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample = cursor.fetchall()
                schema_info[table] = {
                    "columns": [{"name": c[1], "type": c[2]} for c in columns],
                    "sample_rows": sample
                }
            conn.close()
            return {"status": "success", "tables": schema_info}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif tool_name == "run_db_query":
        filepath = os.path.join(workspace, tool_args["path"])
        try:
            conn = sqlite3.connect(filepath)
            cursor = conn.cursor()
            cursor.execute(tool_args["query"])
            results = cursor.fetchall()
            conn.commit()
            conn.close()
            return {"status": "success", "results": results[:20]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif tool_name == "search_code":
        try:
            from rag_store import SimpleRAG
            rag = SimpleRAG(workspace)
            return {"status": "success", "result": rag.search(tool_args.get("query"))}
        except Exception as e:
            return {"status": "error", "message": f"RAG error: {str(e)}"}

    elif tool_name == "index_workspace":
        try:
            from rag_store import SimpleRAG
            rag = SimpleRAG(workspace)
            return {"status": "success", "result": rag.index_workspace()}
        except Exception as e:
            return {"status": "error", "message": f"RAG indexing error: {str(e)}"}

    elif tool_name == "git_status":
        try:
            result = subprocess.run(["git", "status"], cwd=workspace, capture_output=True, text=True, check=True)
            return {"status": "success", "output": result.stdout}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": e.stderr if e.stderr else e.stdout}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif tool_name == "git_diff":
        try:
            cmd = ["git", "diff"]
            if "filepath" in tool_args and tool_args["filepath"]:
                cmd.append(tool_args["filepath"])
            result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, check=True)
            return {"status": "success", "output": result.stdout if result.stdout else "No changes."}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": e.stderr if e.stderr else e.stdout}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif tool_name == "write_file":
        return {
            "status": "needs_permission",
            "type": "write_file",
            "path": tool_args["path"],
            "content": tool_args["content"]
        }

    elif tool_name == "run_command":
        return {
            "status": "needs_permission",
            "type": "run_command",
            "command": tool_args["command"],
            "reason": tool_args.get("reason", "")
        }

    elif tool_name == "ask_user":
        return {
            "status": "needs_permission",
            "type": "ask_user",
            "question": tool_args["question"]
        }

    return {"status": "error", "message": f"Unknown tool: {tool_name}"}


# ─── LLM Call with Provider Fallback ─────────────────────────────────────────

def call_llm(messages, use_tools=True):
    """Call LLM with automatic provider fallback."""
    if not PROVIDERS:
        raise Exception("No LLM providers configured! Set GEMINI_API_KEY or GROQ_API_KEY in .env")

    for provider in PROVIDERS:
        try:
            kwargs = {
                "model": provider["model"],
                "messages": messages,
                "max_tokens": provider["max_tokens"],
                "stream": True,
            }
            if use_tools:
                kwargs["tools"] = TOOLS
                kwargs["tool_choice"] = "auto"

            print(f"  Using {provider['name']} ({provider['model']})")

            # Try streaming first; if it fails (Gemini index bug), retry non-streaming
            try:
                stream = provider["client"].chat.completions.create(**kwargs)
                return stream, provider["name"]
            except Exception as stream_err:
                if 'index' in str(stream_err).lower() or 'key' in str(stream_err).lower():
                    print(f"  Streaming failed, retrying without stream...")
                    kwargs["stream"] = False
                    result = provider["client"].chat.completions.create(**kwargs)
                    return _wrap_as_stream(result), provider["name"]
                raise stream_err

        except Exception as e:
            err_str = str(e)
            if '429' in err_str:
                print(f"  ⚠️  {provider['name']} exhausted, trying next provider...")
            else:
                print(f"  ❌ {provider['name']} error: {err_str[:200]}")

    raise Exception("All LLM providers failed. Check your API keys and rate limits.")


class _FakeChunk:
    """Wraps a non-streaming response to look like a streaming chunk."""
    def __init__(self, choice):
        self.choices = [self]
        self.delta = self
        self.finish_reason = choice.finish_reason
        self.content = choice.message.content
        self.tool_calls = None
        # Convert tool_calls to delta format
        if choice.message.tool_calls:
            self.tool_calls = []
            for i, tc in enumerate(choice.message.tool_calls):
                fake_tc = type('TC', (), {
                    'index': i,
                    'id': tc.id,
                    'function': type('F', (), {
                        'name': tc.function.name,
                        'arguments': tc.function.arguments
                    })()
                })()
                self.tool_calls.append(fake_tc)


def _wrap_as_stream(response):
    """Wrap a non-streaming ChatCompletion response as a single-chunk iterable."""
    for choice in response.choices:
        yield _FakeChunk(choice)


# ─── Message Trimming ────────────────────────────────────────────────────────

def trim_messages(messages):
    """Keep conversation lean — compress old tool results, drop old context."""
    if len(messages) <= 14:
        return messages

    # Keep system prompt + last 12 messages
    trimmed = [messages[0]]  # system prompt

    for msg in messages[-12:]:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if len(content) > 500:
                msg = dict(msg)
                msg["content"] = content[:500] + "\n...(compressed)"
        trimmed.append(msg)

    return trimmed


# ─── Agent Streaming Loop ────────────────────────────────────────────────────

def run_agent_stream(messages, workspace, session_id):
    """Generator that runs the agentic loop and yields SSE events."""

    def generate():
        start_time = time.time()
        stop_flags[session_id] = False
        current_messages = trim_messages(messages)

        for step in range(25):  # max 25 tool rounds (self-correction needs more)
            if stop_flags.get(session_id):
                yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                return

            # ── Inline provider selection with live status feedback ──────────
            response = None
            provider_name = None

            global last_provider_idx
            if 'last_provider_idx' not in globals():
                last_provider_idx = 0

            # Start from the last successful provider, then wrap around
            ordered_providers = [(i, PROVIDERS[i]) for i in range(last_provider_idx, len(PROVIDERS))] + \
                                [(i, PROVIDERS[i]) for i in range(0, last_provider_idx)]

            for idx, provider in ordered_providers:
                try:
                    status_msg = f"Using {provider['name']}…"
                    yield f"data: {json.dumps({'type': 'status', 'text': status_msg})}\n\n"
                    print(f"  {status_msg}")

                    kwargs = {
                        "model": provider["model"],
                        "messages": current_messages,
                        "max_tokens": provider["max_tokens"],
                        "stream": True,
                        "tools": TOOLS,
                        "tool_choice": "auto",
                    }
                    try:
                        stream = provider["client"].chat.completions.create(**kwargs)
                        response = stream
                        provider_name = provider["name"]
                    except Exception as stream_err:
                        err_str_lower = str(stream_err).lower()
                        is_tool_err = any(k in err_str_lower for k in [
                            'index', 'key', 'failed to call a function', 
                            'tool call validation failed', 'tool_use_failed', '400'
                        ])
                        
                        if is_tool_err:
                            # Groq throws these if it hallucinates a tool call or breaks format
                            print(f"  Fallback to non-streaming or no-tools due to: {stream_err}")
                            kwargs["stream"] = False
                            # Always strip tools on 400 fallback since it's almost always a tool syntax hallucination
                            kwargs.pop("tools", None)
                            kwargs.pop("tool_choice", None)
                            
                            try:
                                result = provider["client"].chat.completions.create(**kwargs)
                                response = _wrap_as_stream(result)
                                provider_name = provider["name"]
                            except Exception as fallback_err:
                                raise fallback_err
                        else:
                            raise stream_err
                    
                    last_provider_idx = idx  # Remember successful provider!
                    break  # provider succeeded

                except Exception as e:
                    err_str = str(e)
                    if '429' in err_str:
                        msg = f"⚠️ {provider['name']} exhausted — trying next provider…"
                        print(f"  {msg}")
                        yield f"data: {json.dumps({'type': 'status', 'text': msg})}\n\n"
                    else:
                        msg = f"❌ {provider['name']} error — trying next…"
                        print(f"  {msg}: {err_str[:120]}")
                        yield f"data: {json.dumps({'type': 'status', 'text': msg})}\n\n"

            if not response:
                yield f"data: {json.dumps({'type': 'error', 'text': 'All LLM providers failed. Check your API keys and rate limits.'})}  \n\n"
                return

            # Tell UI which LLM is responding
            yield f"data: {json.dumps({'type': 'provider', 'name': provider_name})}\n\n"

            # Accumulate streaming response
            full_text = ""
            tool_calls_data = {}
            finish_reason = None
            prefix_stripped = False  # strip leading ", " Groq/Llama artifact

            try:
                for chunk in response:
                    if stop_flags.get(session_id):
                        yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                        return

                    choice = chunk.choices[0]
                    finish_reason = choice.finish_reason

                    if choice.delta.content:
                        tok = choice.delta.content
                        # *** BUG FIX: Groq Llama 3.3 70B with tool_choice=auto
                        # sometimes prefixes responses with ", " as a JSON artifact.
                        # Strip any leading ", " / "," from the very first real token.
                        if not prefix_stripped:
                            tok = tok.lstrip(', \n\r')
                            if tok:
                                prefix_stripped = True
                            else:
                                continue  # skip fully-whitespace/comma first tokens
                        full_text += tok
                        yield f"data: {json.dumps({'type': 'token', 'text': tok})}\n\n"

                    if choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            idx = tc.index if hasattr(tc, 'index') and tc.index is not None else 0
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {"id": "", "name": "", "args": ""}
                            if tc.id:
                                tool_calls_data[idx]["id"] = tc.id
                            if tc.function and tc.function.name:
                                tool_calls_data[idx]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls_data[idx]["args"] += tc.function.arguments
            except Exception as stream_err:
                print(f"  Stream error: {stream_err}")
                yield f"data: {json.dumps({'type': 'status', 'text': f'Model error during streaming: {str(stream_err)[:100]} — Auto-retrying...'})}\n\n"
                current_messages.append({
                    "role": "user",
                    "content": f"Your last response failed mid-stream with error: {str(stream_err)}. This usually means you hallucinated a tool call or provided invalid JSON arguments. Please try again and ensure you use the exact tool schema."
                })
                continue

            # Build assistant message for history
            assistant_msg = {"role": "assistant", "content": full_text if full_text else None}
            if tool_calls_data:
                assistant_msg["tool_calls"] = [
                    {
                        "id": v["id"],
                        "type": "function",
                        "function": {
                            "name": v["name"],
                            "arguments": v["args"] if v["args"] else "{}"
                        }
                    }
                    for v in tool_calls_data.values() if v["name"]
                ]
            current_messages.append(assistant_msg)

            # No tool calls — agent is done talking
            if not tool_calls_data or finish_reason == "stop":
                elapsed = time.time() - start_time
                yield f"data: {json.dumps({'type': 'done', 'elapsed': elapsed})}\n\n"
                conversation_history[session_id] = current_messages
                return

            # Execute each tool call
            for idx, tc in sorted(tool_calls_data.items()):
                if not tc["name"]:
                    continue

                tool_name = tc["name"]
                try:
                    tool_args = json.loads(tc["args"]) if tc["args"] else {}
                except json.JSONDecodeError:
                    tool_args = {}

                print(f"  🔧 Tool: {tool_name} | Args: {tool_args}")
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': tool_args})}\n\n"

                result = execute_tool(tool_name, tool_args, workspace)

                # If tool needs user permission, pause the loop
                if result.get("status") == "needs_permission":
                    conversation_history[session_id] = current_messages
                    yield f"data: {json.dumps({'type': 'needs_permission', 'tool_call_id': tc['id'], 'tool': tool_name, 'result': result})}\n\n"
                    return

                # Add tool result to conversation
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result)
                })
                yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': result})}\n\n"

            # Trim after each tool round
            current_messages = trim_messages(current_messages)

        elapsed = time.time() - start_time
        yield f"data: {json.dumps({'type': 'done', 'elapsed': elapsed})}\n\n"
        conversation_history[session_id] = current_messages

    return generate


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "providers": [p["name"] for p in PROVIDERS]
    })


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    data = request.json
    user_message = data.get('message', '')
    workspace = data.get('workspace', '.')
    session_id = data.get('session_id', 'default')
    context = data.get('context', {})

    print(f"\n\U0001f464 User: {user_message}")
    print(f"\U0001f4c1 Workspace: {workspace}")
    if context.get('active_file'):
        print(f"\U0001f441\ufe0f  Active: {context['active_file']} (line {context.get('cursor_line', '?')})")

    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # Build context block to inject into the user message
    context_parts = []
    if context.get('active_file'):
        context_parts.append(f"Active file: {context['active_file']} (line {context.get('cursor_line', '?')}, language: {context.get('active_file_language', 'unknown')})")
    if context.get('selected_text'):
        context_parts.append(f"Selected text:\n```\n{context['selected_text']}\n```")
    if context.get('open_files'):
        context_parts.append(f"Open files: {', '.join(context['open_files'])}")
    if context.get('active_file_content'):
        context_parts.append(f"Active file content:\n```{context.get('active_file_language', '')}\n{context['active_file_content']}\n```")

    full_message = user_message
    if context_parts:
        context_block = "[CONTEXT]\n" + "\n\n".join(context_parts) + "\n[/CONTEXT]"
        full_message = f"{context_block}\n\n{user_message}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history[session_id]
    messages.append({"role": "user", "content": full_message})

    return Response(
        stream_with_context(run_agent_stream(messages, workspace, session_id)()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/chat/continue', methods=['POST'])
def chat_continue():
    data = request.json
    session_id = data.get('session_id', 'default')
    tool_call_id = data.get('tool_call_id', '')
    action_result = data.get('action_result', '')
    workspace = data.get('workspace', '.')

    # FIXED: Include system prompt so agent keeps its instructions
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history.get(session_id, [])
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": action_result
    })

    return Response(
        stream_with_context(run_agent_stream(messages, workspace, session_id)()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/chat/stop', methods=['POST'])
def stop():
    data = request.json
    session_id = data.get('session_id', 'default')
    stop_flags[session_id] = True
    return jsonify({"status": "stopping"})


@app.route('/chat/clear', methods=['POST'])
def clear():
    data = request.json
    session_id = data.get('session_id', 'default')
    conversation_history.pop(session_id, None)
    return jsonify({"status": "cleared"})


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  🤖 AI Coding Agent Server")
    print("=" * 55)
    if PROVIDERS:
        for p in PROVIDERS:
            print(f"  ✅ {p['name']} ({p['model']})")
    else:
        print("  ⚠️  No providers! Set GEMINI_API_KEY or GROQ_API_KEY in .env")
    print(f"\n  🚀 Running on http://localhost:5001")
    print("=" * 55)
    app.run(port=5001, debug=False, threaded=True)
