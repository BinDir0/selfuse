#!/usr/bin/env python3
import json
import subprocess
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractionHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, audio_dir=None, **kwargs):
        self.audio_dir = audio_dir
        super().__init__(*args, **kwargs)

    def do_POST(self):
        """处理音频播放请求"""
        if self.path.startswith('/play/'):
            audio_name = self.path[6:]
            self._handle_play(audio_name)
        else:
            self.send_error(404)

    def do_GET(self):
        """处理阻塞式输入请求"""
        if self.path == '/get_input':
            self._handle_terminal_input()
        else:
            self.send_error(404)

    def _handle_play(self, name):
        for ext in ['.mp3', '.wav']:
            file_path = self.audio_dir / f"{name}{ext}"
            if file_path.exists():
                logger.info(f"🎵 播放音频: {file_path.name}")
                # 非阻塞播放
                subprocess.Popen(['mpg123', '-q', str(file_path)], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
                self._send_json({"status": "success"})
                return
        self.send_error(404, "Audio file not found")

    def _handle_terminal_input(self):
        """核心：在宿主机终端请求输入，结果返回给 Docker"""
        print("\n" + "="*40)
        print("🤖 机器人交互提示")
        instr = input("[Input] 请输入语言指令: ").strip()
        mode = input("[Input] 请输入模式 (deploy/debug) [deploy]: ").strip().lower()
        if not mode:
            mode = "deploy"
        print("="*40 + "\n")
        
        self._send_json({
            "instruction": instr,
            "mode": mode
        })

    def _send_json(self, data):
        response_data = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_data)))
        self.end_headers()
        self.wfile.write(response_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--audio-dir', default='./audio')
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).absolute()
    server = HTTPServer(('0.0.0.0', args.port), 
                        lambda *a, **k: InteractionHandler(*a, audio_dir=audio_dir, **k))
    
    logger.info(f"🚀 宿主机交互服务已启动: http://localhost:{args.port}")
    logger.info(f"音频目录: {audio_dir}")
    server.serve_forever()

if __name__ == '__main__':
    main()