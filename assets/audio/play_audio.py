#!/usr/bin/env python3
import os
import json
import threading
import subprocess
import argparse
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioPlayHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, audio_dir=None, **kwargs):
        self.audio_dir = audio_dir
        super().__init__(*args, **kwargs)

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith('/play/'):
            audio_name = path[6:]
            self._handle_play(audio_name)
        else:
            self.send_error(404)

    def _handle_play(self, name):
        # 支持多种后缀
        for ext in ['.mp3', '.wav', '.ogg']:
            file_path = self.audio_dir / f"{name}{ext}"
            if file_path.exists():
                logger.info(f"🎵 Playing: {file_path.name}")
                threading.Thread(target=lambda: subprocess.run(['mpg123', '-q', str(file_path)]), daemon=True).start()
                self._send_json({"status": "success", "file": name})
                return
        self.send_error(404, f"File {name} not found")

    def _send_json(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--audio-dir', default='.') # 指向包含那4个mp3的目录
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).absolute()
    server = HTTPServer(('0.0.0.0', args.port), lambda *a, **k: AudioPlayHandler(*a, audio_dir=audio_dir, **k))
    logger.info(f"🚀 Audio Server on http://0.0.0.0:{args.port}, Dir: {audio_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()

if __name__ == '__main__':
    main()