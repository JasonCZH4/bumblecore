import argparse
import uuid
import json
from typing import Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
# ======================
# 全局状态
# ======================
import os

# 设置只使用第 2 张物理显卡（编号为 1）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

bot = None
training_stage = "sft"
enable_history_global = False
sessions: Dict[str, list] = {}

def load_model(model_path: str, device_map: str, dtype: str, stage: str, enable_history: bool):
    global bot, training_stage, enable_history_global
    print(f"正在加载模型: {model_path}...")

    from bumblecore.inference import BumblebeeChat
    bot = BumblebeeChat(
        model_path=model_path,
        device_map=device_map,
        dtype=dtype
    )
    
    training_stage = stage
    enable_history_global = enable_history
    status = "启用" if enable_history_global else "禁用"
    print(f"✅ 模型加载完成！训练阶段: {training_stage}，对话历史: {status}")

app = FastAPI(title="Bumblebee Chat with Web UI")

# 添加CORS中间件，方便调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# API Models
# ======================

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    max_new_tokens: int = 512
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    do_sample: bool = True

# ======================
# SSE Stream Generator (修复版本)
# ======================

def _make_sse(data: dict) -> str:
    sse = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    # 添加 padding 注释，确保 > 1KB
    if len(sse.encode()) < 1024:
        pad_len = 1024 - len(sse.encode()) + 10
        sse += ":" + " " * pad_len + "\n\n"
    return sse

async def chat_stream_generator(session_id: str, request: ChatRequest) -> AsyncGenerator[str, None]:
    global bot, training_stage, enable_history_global, sessions
    
    if bot is None:
        yield _make_sse({'error': '模型未加载'})
        return

    user_input = request.message.strip()
    if not user_input:
        yield _make_sse({'error': '消息不能为空'})
        return

    gen_kwargs = {}
    if request.temperature is not None: gen_kwargs["temperature"] = request.temperature
    if request.top_k is not None: gen_kwargs["top_k"] = request.top_k
    if request.top_p is not None: gen_kwargs["top_p"] = request.top_p
    if request.repetition_penalty is not None: gen_kwargs["repetition_penalty"] = request.repetition_penalty
    gen_kwargs["do_sample"] = request.do_sample
    gen_kwargs["max_new_tokens"] = request.max_new_tokens

    full_response = ""
    try:
        if training_stage == "pretrain":
            messages = user_input
            stream = bot.stream_chat(messages=messages, **gen_kwargs)
        else:
            if enable_history_global:
                if session_id not in sessions:
                    sessions[session_id] = []
                messages = sessions[session_id] + [{"role": "user", "content": user_input}]
            else:
                messages = [{"role": "user", "content": user_input}]
            stream = bot.stream_chat(messages=messages, system_prompt=request.system_prompt, **gen_kwargs)

        for token in stream:
            if token:
                yield _make_sse({'token': token})
                full_response += token
                await asyncio.sleep(0)  # 让出控制权

        yield _make_sse({'done': True})

        if training_stage != "pretrain" and enable_history_global:
            sessions.setdefault(session_id, [])
            sessions[session_id].append({"role": "user", "content": user_input})
            sessions[session_id].append({"role": "assistant", "content": full_response})

    except Exception as e:
        print(f"生成错误: {e}")
        yield _make_sse({'error': str(e)})

# ======================
# API Routes
# ======================

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, session_id: Optional[str] = Query(None)):
    sid = session_id or str(uuid.uuid4())
    print(f"开始流式对话，session_id: {sid}")
    return StreamingResponse(
        chat_stream_generator(sid, request),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # 防止nginx等代理缓冲
        }
    )

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": "会话已清除"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": bot is not None}

# ======================
# 修复后的前端页面
# ======================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Bumblebee Chat</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }
            #chat-box {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                height: 500px;
                overflow-y: auto;
                background: white;
                margin-bottom: 15px;
            }
            .message {
                margin-bottom: 12px;
                line-height: 1.5;
                padding: 8px 12px;
                border-radius: 6px;
            }
            .user { 
                background-color: #e3f2fd; 
                margin-left: 20%;
                margin-right: 0;
            }
            .bot { 
                background-color: #f1f8e9; 
                margin-right: 20%;
                margin-left: 0;
            }
            .thinking { color: #6c757d; font-style: italic; }
            .input-area {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }
            #user-input {
                flex: 1;
                padding: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 16px;
            }
            button {
                padding: 12px 20px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover { background: #0056b3; }
            button:disabled { background: #6c757d; cursor: not-allowed; }
            .status {
                margin-bottom: 10px;
                color: #666;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <h1>🐝 Bumblebee Chat</h1>
        <div class="status">Session ID: <span id="session-id"></span></div>
        <div id="chat-box"></div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="输入消息..." />
            <button id="send-btn">发送</button>
            <button id="clear-btn">清空会话</button>
        </div>

        <script>
            const chatBox = document.getElementById('chat-box');
            const userInput = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');
            const clearBtn = document.getElementById('clear-btn');
            const sessionIdSpan = document.getElementById('session-id');

            // 生成或获取session ID
            let sessionId = localStorage.getItem('sessionId');
            if (!sessionId) {
                sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('sessionId', sessionId);
            }
            sessionIdSpan.textContent = sessionId;

            function addMessage(sender, text, isThinking = false) {
                const div = document.createElement('div');
                div.className = `message ${sender}`;
                if (isThinking) {
                    div.classList.add('thinking');
                }
                const prefix = sender === 'user' ? '👤 用户:' : '🤖 助手:';
                div.innerHTML = `<strong>${prefix}</strong> ${text}`;
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
                return div;
            }

            async function sendMessage() {
                const msg = userInput.value.trim();
                if (!msg || sendBtn.disabled) return;

                // 添加用户消息
                addMessage('user', msg);
                userInput.value = '';
                sendBtn.disabled = true;

                // 添加正在生成的占位符
                const thinkingDiv = addMessage('bot', '正在思考...', true);
                const botResponseSpan = document.createElement('span');
                thinkingDiv.innerHTML = '<strong>🤖 助手:</strong> ';
                thinkingDiv.appendChild(botResponseSpan);

                try {
                    const response = await fetch('/chat/stream?session_id=' + encodeURIComponent(sessionId), {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: msg,
                            max_new_tokens: 512,
                            temperature: 0.7,
                            do_sample: true
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder('utf-8');
                    let buffer = '';
                    let accumulatedText = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\\n');
                        
                        // 保留最后一行（可能不完整）
                        buffer = lines.pop() || '';

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const dataStr = line.slice(6);
                                if (dataStr.trim() === '') continue;
                                
                                try {
                                    const data = JSON.parse(dataStr);
                                    
                                    if (data.token !== undefined) {
                                        accumulatedText += data.token;
                                        botResponseSpan.textContent = accumulatedText;
                                        chatBox.scrollTop = chatBox.scrollHeight;
                                    } else if (data.done) {
                                        console.log('Stream completed');
                                        // 移除thinking样式
                                        thinkingDiv.classList.remove('thinking');
                                    } else if (data.error) {
                                        botResponseSpan.textContent = '错误: ' + data.error;
                                        thinkingDiv.classList.remove('thinking');
                                        break;
                                    }
                                } catch (e) {
                                    console.error('Parse error:', e, 'Data:', dataStr);
                                }
                            }
                        }
                    }

                    // 处理buffer中剩余的数据
                    if (buffer.trim()) {
                        const lines = buffer.split('\\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const dataStr = line.slice(6);
                                try {
                                    const data = JSON.parse(dataStr);
                                    if (data.token) {
                                        accumulatedText += data.token;
                                        botResponseSpan.textContent = accumulatedText;
                                    }
                                } catch (e) {
                                    console.error('Parse error on buffer:', e);
                                }
                            }
                        }
                    }

                    // 移除thinking样式
                    thinkingDiv.classList.remove('thinking');

                } catch (error) {
                    console.error('Error:', error);
                    botResponseSpan.textContent = '请求失败: ' + error.message;
                    thinkingDiv.classList.remove('thinking');
                } finally {
                    sendBtn.disabled = false;
                    userInput.focus();
                }
            }

            function clearChat() {
                fetch(`/session/${sessionId}`, {
                    method: 'DELETE'
                }).then(() => {
                    chatBox.innerHTML = '';
                    // 生成新的session ID
                    sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                    localStorage.setItem('sessionId', sessionId);
                    sessionIdSpan.textContent = sessionId;
                }).catch(console.error);
            }

            sendBtn.addEventListener('click', sendMessage);
            clearBtn.addEventListener('click', clearChat);
            
            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            userInput.focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ======================
# 主入口
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动 Bumblebee Chat Web 服务")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--training_stage", type=str, required=True,
                        choices=["sft", "dpo", "pretrain"])
    parser.add_argument("--enable_history", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    # 加载模型
    load_model(
        model_path=args.model_path,
        device_map=args.device_map,
        dtype=args.dtype,
        stage=args.training_stage,
        enable_history=args.enable_history
    )

    # 启动服务
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )