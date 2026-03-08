#!/usr/bin/env python3
import json
import threading
import subprocess
import argparse
import sys
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import logging
from urllib.parse import parse_qs

# ==========================================
# 状态管理
# ==========================================
class GlobalState:
    def __init__(self):
        self.is_robot_waiting = False  # 推理程序是否正在等待指令
        self.current_command = None    # 存放当前输入的指令
        self.command_event = threading.Event() 
        self.server_ready = False      

state = GlobalState()

# ==========================================
# 日志处理：自定义处理器，确保日志和状态指示器不冲突
# ==========================================
class CleanLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # \r: 回到行首, \033[K: 清除当前行
            sys.stdout.write('\r\033[K' + msg + '\n')
            sys.stdout.flush()
        except Exception:
            self.handleError(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
for h in logger.handlers[:]:
    logger.removeHandler(h)
handler = CleanLoggingHandler()
handler.setFormatter(logging.Formatter('\033[94m%(asctime)s\033[0m - %(message)s'))
logger.addHandler(handler)

# ==========================================
# 网页 HTML (完美交互版)
# ==========================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legendary VLA Control Panel</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 550px; margin: 40px auto; padding: 20px; transition: background 0.4s; background: #f4f7f9; }
        .card { background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); text-align: center; }
        .status-idle { background: #f0f2f5; }
        .status-active { background: #e3f2fd; }
        h1 { margin-bottom: 25px; color: #2c3e50; font-size: 24px; }
        .mode-container { display: flex; justify-content: center; gap: 15px; margin-bottom: 20px; }
        .mode-option { flex: 1; position: relative; }
        .mode-option input { display: none; }
        .mode-label { display: block; padding: 10px; border: 2px solid #ddd; border-radius: 10px; cursor: pointer; font-weight: bold; color: #7f8c8d; transition: all 0.3s; }
        .mode-option input:checked + .mode-label { border-color: #3498db; background: #ebf5fb; color: #3498db; }
        .mode-option input:disabled + .mode-label { opacity: 0.5; cursor: not-allowed; background: #f5f5f5; border-color: #eee; color: #ccc; }
        textarea { width: 100%; padding: 15px; margin-bottom: 20px; border: 2px solid #ddd; border-radius: 10px; font-size: 16px; box-sizing: border-box; outline: none; transition: border 0.3s; resize: none; overflow-y: hidden; font-family: inherit; line-height: 1.5; min-height: 54px; }
        textarea:focus { border-color: #3498db; }
        textarea:disabled { background: #f9f9f9; cursor: not-allowed; }
        button { width: 100%; padding: 16px; background: #3498db; color: white; border: none; border-radius: 10px; font-size: 18px; cursor: pointer; font-weight: bold; transition: background 0.3s; }
        button:disabled { background: #bdc3c7; cursor: not-allowed; }
        #indicator { font-weight: bold; margin-bottom: 20px; padding: 12px; border-radius: 10px; font-size: 14px; transition: all 0.3s; }
        .idle-msg { color: #7f8c8d; background: #e9ecef; border: 1px solid #dee2e6; }
        .active-msg { color: #2980b9; background: #d1ecf1; border: 1px solid #3498db; box-shadow: 0 0 15px rgba(52,152,219,0.2); }
    </style>
</head>
<body class="status-idle">
    <div class="card">
        <div id="indicator" class="idle-msg">☁️ 当前推理程序未请求用户输入，等待请求...</div>
        <h1>🤖 Legendary VLA 指令终端</h1>
        <form id="cmdForm">
            <div class="mode-container">
                <label class="mode-option"><input type="radio" name="mode" value="deploy" checked id="m1" disabled><span class="mode-label">Deploy 部署</span></label>
                <label class="mode-option"><input type="radio" name="mode" value="debug" id="m2" disabled><span class="mode-label">Debug 调试</span></label>
            </div>
            <textarea id="instruction" name="instruction" placeholder="..." disabled autocomplete="off" rows="1"></textarea>
            <button type="submit" id="submitBtn" disabled>确认发送指令</button>
        </form>
    </div>
    <script>
        const form = document.getElementById('cmdForm'), input = document.getElementById('instruction'), btn = document.getElementById('submitBtn'), indicator = document.getElementById('indicator'), modes = document.querySelectorAll('input[name="mode"]'), body = document.body;
        input.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = (this.scrollHeight) + 'px'; });
        input.addEventListener('keydown', function(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if(!btn.disabled) form.requestSubmit(); } });
        async function checkStatus() {
            try {
                const resp = await fetch('/status'), data = await resp.json();
                if (data.waiting) {
                    if (input.disabled) { input.disabled = false; btn.disabled = false; modes.forEach(m => m.disabled = false); input.placeholder = "请您输入指令..."; indicator.innerText = "🎯 推理程序正在请求输入，请回复"; indicator.className = "active-msg"; body.className = "status-active"; input.focus(); }
                } else {
                    if (!input.disabled) { input.disabled = true; btn.disabled = true; modes.forEach(m => m.disabled = true); input.value = ""; input.style.height = 'auto'; input.placeholder = "等待请求中..."; indicator.innerText = "☁️ 当前推理程序未请求用户输入，等待请求..."; indicator.className = "idle-msg"; body.className = "status-idle"; }
                }
            } catch (e) {}
        }
        setInterval(checkStatus, 500);
        form.onsubmit = async (e) => { e.preventDefault(); const val = input.value.trim(); if(!val) return; btn.disabled = true; await fetch('/web_submit', { method: 'POST', body: new URLSearchParams(new FormData(form)) }); input.value = ""; input.style.height = 'auto'; };
    </script>
</body>
</html>
"""

# ==========================================
# Server 核心逻辑
# ==========================================

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class InteractionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): return

    def do_GET(self):
        if self.path == '/': self._send_html()
        elif self.path == '/status': self._send_json({"waiting": state.is_robot_waiting})
        elif self.path == '/get_input': self._handle_robot_request()
        else: self.send_error(404)

    def do_POST(self):
        if self.path == '/web_submit': self._handle_web_submit()
        elif self.path.startswith('/play/'): self._handle_play(self.path[6:])
        else: self.send_error(404)

    def _handle_robot_request(self):
        logger.info("📡 接收到推理程序请求: 已开启网页端输入通道")
        state.command_event.clear()
        state.is_robot_waiting = True
        
        # 阻塞直到网页端提交
        state.command_event.wait(timeout=None) 
        
        if state.current_command:
            try:
                self._send_json(state.current_command)
                logger.info(f"✅ 指令送达: '{state.current_command['instruction']}' ({state.current_command['mode']})")
            except Exception as e:
                logger.warning(f"⚠️ 发送失败: {e}")
        
        state.is_robot_waiting = False
        state.current_command = None

    def _handle_web_submit(self):
        content_length = int(self.headers['Content-Length'])
        params = parse_qs(self.rfile.read(content_length).decode('utf-8'))
        instr = params.get('instruction', [''])[0].strip()
        mode = params.get('mode', ['deploy'])[0].strip()
        if instr and state.is_robot_waiting:
            state.current_command = {"instruction": instr, "mode": mode}
            state.command_event.set()
            self.send_response(200); self.end_headers()

    def _handle_play(self, name):
        audio_dir = Path("./audio")
        for ext in ['.mp3', '.wav']:
            file_path = audio_dir / f"{name}{ext}"
            if file_path.exists():
                subprocess.Popen(['mpg123', '-q', str(file_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
        self.send_response(200); self.end_headers()

    def _send_json(self, data):
        response = json.dumps(data).encode('utf-8')
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(response)

    def _send_html(self):
        self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.end_headers()
        self.wfile.write(HTML_PAGE.encode('utf-8'))

# ==========================================
# 状态监听线程：仅负责在终端显示当前状态
# ==========================================
def terminal_status_thread():
    while not state.server_ready:
        time.sleep(0.1)
    
    while True:
        if state.is_robot_waiting:
            sys.stdout.write("\r\033[K🎯 推理程序正在请求输入，请在网页端操作...")
        else:
            sys.stdout.write("\r\033[K☁️ 当前推理程序未请求用户输入，等待请求...")
        sys.stdout.flush()
        time.sleep(0.5)

def main():
    # 启动状态刷新线程
    threading.Thread(target=terminal_status_thread, daemon=True).start()

    port = 8080
    print("\n" + "="*55)
    print(f"🚀 Legendary VLA 交互服务已就绪")
    print(f"🔗 控制面板: http://localhost:{port}")
    print(f"🔗 局域网访问: http://<宿主机IP>:{port}")
    print("="*55 + "\n")
    
    time.sleep(0.2)
    state.server_ready = True

    server = ThreadingHTTPServer(('0.0.0.0', port), InteractionHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已关闭")
        sys.exit(0)

if __name__ == '__main__':
    main()