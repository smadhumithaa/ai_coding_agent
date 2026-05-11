import * as vscode from 'vscode';
import * as http from 'http';
import * as fs from 'fs';
import * as path from 'path';
import * as cp from 'child_process';

const SERVER_PORT = 5001;

export function activate(context: vscode.ExtensionContext) {
    const provider = new ChatViewProvider(context);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('aiCodingAgent.chatView', provider)
    );
}

class ChatViewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private _terminal?: vscode.Terminal;
    private _msgHandlerDisposable?: vscode.Disposable;  // prevent duplicate handlers

    constructor(private readonly _context: vscode.ExtensionContext) {}

    private _getOrCreateTerminal(): vscode.Terminal {
        if (this._terminal && !this._terminal.exitStatus) {
            return this._terminal;
        }
        this._terminal = vscode.window.createTerminal({
            name: 'AI Agent',
            cwd: vscode.workspace.workspaceFolders?.[0]?.uri?.fsPath || '.'
        });
        this._terminal.show();
        return this._terminal;
    }

    resolveWebviewView(webviewView: vscode.WebviewView) {
        this._view = webviewView;
        webviewView.webview.options = { enableScripts: true };

        const htmlPath = path.join(this._context.extensionPath, 'src', 'webview.html');
        webviewView.webview.html = fs.readFileSync(htmlPath, 'utf8');

        // *** BUG FIX: dispose previous handler before registering a new one.
        // resolveWebviewView() is called every time the sidebar panel becomes visible,
        // which stacks duplicate onDidReceiveMessage handlers causing double-responses.
        this._msgHandlerDisposable?.dispose();
        this._msgHandlerDisposable = webviewView.webview.onDidReceiveMessage(async (message) => {

            if (message.command === 'sendMessage') {
                const workspacePath = vscode.workspace.workspaceFolders?.[0]?.uri?.fsPath || '.';
                const context = this._gatherContext();
                this._streamFromAgent('/chat/stream', {
                    message: message.text,
                    workspace: workspacePath,
                    session_id: 'vscode-session',
                    context
                });
            }

            if (message.command === 'continueAgent') {
                const workspacePath = vscode.workspace.workspaceFolders?.[0]?.uri?.fsPath || '.';
                const context = this._gatherContext();
                this._streamFromAgent('/chat/continue', {
                    session_id: 'vscode-session',
                    action_result: message.result,
                    tool_call_id: message.toolCallId,
                    workspace: workspacePath,
                    context
                });
            }

            if (message.command === 'stopAgent') {
                this._postToAgent('/chat/stop', { session_id: 'vscode-session' });
            }

            if (message.command === 'clearHistory') {
                this._postToAgent('/chat/clear', { session_id: 'vscode-session' });
            }

            if (message.command === 'runApproved') {
                const workspacePath = vscode.workspace.workspaceFolders?.[0]?.uri?.fsPath || '.';
                await this._runCommandAndWait(message.text, workspacePath, message.toolCallId);
            }

            if (message.command === 'writeApproved') {
                try {
                    const workspacePath = vscode.workspace.workspaceFolders?.[0]?.uri?.fsPath || '.';
                    const fullPath = path.join(workspacePath, message.filePath);
                    const dir = path.dirname(fullPath);
                    if (!fs.existsSync(dir)) {
                        fs.mkdirSync(dir, { recursive: true });
                    }

                    // Show diff if file exists
                    if (fs.existsSync(fullPath)) {
                        const oldContent = fs.readFileSync(fullPath, 'utf8');
                        const oldUri = vscode.Uri.parse(`untitled:${message.filePath} (original)`);
                        const newUri = vscode.Uri.parse(`untitled:${message.filePath} (new)`);
                        // Write new content to a temp file for diffing
                        const tmpPath = fullPath + '.aicoder.tmp';
                        fs.writeFileSync(tmpPath, message.content, 'utf8');
                        await vscode.commands.executeCommand(
                            'vscode.diff',
                            vscode.Uri.file(fullPath),
                            vscode.Uri.file(tmpPath),
                            `AI Edit: ${message.filePath}`
                        );
                        // Clean up tmp after a short delay
                        setTimeout(() => {
                            try { fs.unlinkSync(tmpPath); } catch {}
                        }, 30000);
                    }

                    fs.writeFileSync(fullPath, message.content, 'utf8');
                    const doc = await vscode.workspace.openTextDocument(fullPath);
                    await vscode.window.showTextDocument(doc, { preview: false });
                    webviewView.webview.postMessage({
                        command: 'actionDone',
                        success: true,
                        result: `Successfully wrote ${message.filePath}`,
                        toolCallId: message.toolCallId
                    });
                } catch (e: any) {
                    webviewView.webview.postMessage({
                        command: 'actionDone',
                        success: false,
                        result: `Error writing file: ${e.message}`,
                        toolCallId: message.toolCallId
                    });
                }
            }
        });
    }

    /** Gather active editor context to send with every message */
    private _gatherContext(): object {
        const editor = vscode.window.activeTextEditor;
        const openFiles = vscode.window.visibleTextEditors
            .map(e => vscode.workspace.asRelativePath(e.document.uri))
            .filter(p => !p.startsWith('untitled'));

        if (!editor) {
            return { open_files: openFiles };
        }

        const doc = editor.document;
        const relativePath = vscode.workspace.asRelativePath(doc.uri);
        const selection = editor.selection;
        const selectedText = !selection.isEmpty ? doc.getText(selection) : '';

        // Only send file content if it's reasonably sized
        const content = doc.getText();
        const truncated = content.length > 6000
            ? content.substring(0, 6000) + '\n...(file truncated for context)'
            : content;

        return {
            active_file: relativePath,
            active_file_content: truncated,
            active_file_language: doc.languageId,
            cursor_line: selection.active.line + 1,
            selected_text: selectedText,
            open_files: openFiles
        };
    }

    private async _runCommandAndWait(command: string, workspacePath: string, toolCallId?: string) {
        const terminal = this._getOrCreateTerminal();
        terminal.show();
        terminal.sendText(command);

        this._view?.webview.postMessage({
            command: 'commandRunning',
            text: command
        });

        cp.exec(command, { cwd: workspacePath, timeout: 120000, maxBuffer: 1024 * 1024 * 5, encoding: 'utf8' }, (error, stdout, stderr) => {
            const output = ((stdout || '') + (stderr || '')).trim();
            const hasError = !!error
                || (stderr || '').toLowerCase().includes('error')
                || (stderr || '').toLowerCase().includes('failed')
                || (stderr || '').toLowerCase().includes('traceback');

            const truncated = output.length > 2000
                ? output.substring(0, 2000) + '\n...(truncated)'
                : output;

            this._view?.webview.postMessage({
                command: 'actionDone',
                success: !hasError,
                result: hasError
                    ? `Command failed:\n${truncated || error?.message || 'Unknown error'}`
                    : `Command succeeded:\n${truncated || '(no output)'}`,
                toolCallId: toolCallId
            });
        });
    }

    private _streamFromAgent(endpoint: string, body: object) {
        const bodyStr = JSON.stringify(body);
        const options = {
            hostname: 'localhost',
            port: SERVER_PORT,
            path: endpoint,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(bodyStr)
            }
        };

        const req = http.request(options, (res: any) => {
            let buffer = '';
            res.on('data', (chunk: any) => {
                buffer += chunk.toString();
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(trimmed.slice(6));
                            this._view?.webview.postMessage(data);
                        } catch {}
                    }
                }
            });
            // Flush remaining buffer when stream closes cleanly
            res.on('end', () => {
                const remaining = buffer.trim();
                if (remaining.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(remaining.slice(6));
                        this._view?.webview.postMessage(data);
                    } catch {}
                }
            });
        });

        req.on('error', () => {
            this._view?.webview.postMessage({
                type: 'error',
                text: 'Cannot connect to agent server. Run `python server.py` first!'
            });
        });

        req.write(bodyStr);
        req.end();
    }

    private _postToAgent(endpoint: string, body: object) {
        const bodyStr = JSON.stringify(body);
        const options = {
            hostname: 'localhost',
            port: SERVER_PORT,
            path: endpoint,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(bodyStr)
            }
        };
        const req = http.request(options, () => {});
        req.on('error', () => {});
        req.write(bodyStr);
        req.end();
    }
}

export function deactivate() {}
