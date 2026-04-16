#!/usr/bin/env python3
"""Serve a lightweight dashboard for shared dataset pipeline runs."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.shared_run_monitor import summarize_runs


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shared Run Monitor</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #121a31;
      --panel-2: #0f1730;
      --text: #e9eefc;
      --muted: #9fb0d9;
      --line: #263353;
      --blue: #5da9ff;
      --green: #4dd68c;
      --yellow: #ffcd57;
      --red: #ff6b7d;
      --gray: #7f8aa8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #0b1020 0%, #08101a 100%);
      color: var(--text);
    }
    .wrap {
      max-width: 1800px;
      margin: 0 auto;
      padding: 20px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      margin-bottom: 18px;
    }
    .title {
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .meta {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      background: rgba(18, 26, 49, 0.86);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      margin-bottom: 16px;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    input {
      width: 180px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      padding: 10px 12px;
      font-size: 14px;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .card {
      background: rgba(18, 26, 49, 0.86);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
    }
    .card .k {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 8px;
    }
    .card .v {
      font-size: 24px;
      font-weight: 700;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: rgba(18, 26, 49, 0.86);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(38, 51, 83, 0.7);
      vertical-align: top;
      text-align: left;
      font-size: 14px;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      background: rgba(8, 16, 26, 0.65);
      position: sticky;
      top: 0;
    }
    tbody tr:hover {
      background: rgba(93, 169, 255, 0.06);
    }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.03em;
      margin-right: 6px;
      margin-bottom: 6px;
      white-space: nowrap;
    }
    .health-running { background: rgba(93, 169, 255, 0.16); color: var(--blue); }
    .health-done { background: rgba(77, 214, 140, 0.16); color: var(--green); }
    .health-stalled, .health-failed, .health-error { background: rgba(255, 107, 125, 0.16); color: var(--red); }
    .health-partial { background: rgba(255, 205, 87, 0.16); color: var(--yellow); }
    .health-empty { background: rgba(127, 138, 168, 0.16); color: var(--gray); }
    .src-status { background: rgba(93, 169, 255, 0.12); color: var(--blue); }
    .src-backup { background: rgba(255, 205, 87, 0.12); color: var(--yellow); }
    .src-events, .src-missing, .src-error { background: rgba(255, 107, 125, 0.12); color: var(--red); }
    .stage {
      background: rgba(255, 255, 255, 0.06);
      color: var(--text);
      font-weight: 600;
    }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .small { font-size: 12px; line-height: 1.5; }
    .nowrap { white-space: nowrap; }
    .warnings {
      color: var(--yellow);
      max-width: 420px;
    }
    @media (max-width: 1100px) {
      th:nth-child(7), td:nth-child(7),
      th:nth-child(8), td:nth-child(8) {
        display: none;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <div class="title">Shared Run Monitor</div>
        <div class="meta" id="meta">Loading...</div>
      </div>
    </div>
    <div class="controls">
      <label>Pattern<input id="pattern" value="20*"></label>
      <label>Run Tags<input id="run_tags" placeholder="comma,separated"></label>
      <label>Limit<input id="limit" value="24"></label>
      <label>Stall Sec<input id="stall_seconds" value="1800"></label>
      <label>Refresh Sec<input id="refresh_seconds" value="5"></label>
    </div>
    <div class="summary" id="summary"></div>
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Health</th>
          <th>Progress</th>
          <th>Last Event</th>
          <th>Stages</th>
          <th>Config</th>
          <th>Manifest</th>
          <th>Warnings</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <script>
    const ids = ["pattern", "run_tags", "limit", "stall_seconds", "refresh_seconds"];
    const summaryEl = document.getElementById("summary");
    const rowsEl = document.getElementById("rows");
    const metaEl = document.getElementById("meta");
    let timer = null;

    function esc(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function pill(text, cls) {
      return `<span class="pill ${cls}">${esc(text)}</span>`;
    }

    function updateUrl() {
      const params = new URLSearchParams();
      for (const id of ids) {
        const value = document.getElementById(id).value.trim();
        if (value) params.set(id, value);
      }
      history.replaceState(null, "", `?${params.toString()}`);
      return params;
    }

    function restoreFromUrl() {
      const params = new URLSearchParams(location.search);
      for (const id of ids) {
        if (params.has(id)) document.getElementById(id).value = params.get(id);
      }
    }

    function stagePills(run) {
      return (run.stages || []).map((stage) => {
        const counts = (run.stage_counts || {})[stage] || {};
        const done = counts.completed || 0;
        const fail = counts.failed || 0;
        const pend = counts.pending || 0;
        return pill(`${stage} ${done}/${run.total || 0}${fail ? ` f=${fail}` : ""}${pend ? ` p=${pend}` : ""}`, "stage");
      }).join("");
    }

    function renderSummary(payload) {
      const runs = payload.runs || [];
      const counters = { running: 0, stalled: 0, done: 0, failed: 0, partial: 0, empty: 0, error: 0 };
      for (const run of runs) {
        counters[run.health] = (counters[run.health] || 0) + 1;
      }
      const items = [
        ["Runs", runs.length],
        ["Running", counters.running || 0],
        ["Stalled", counters.stalled || 0],
        ["Done", counters.done || 0],
        ["Failed", counters.failed || 0],
        ["Partial", counters.partial || 0],
      ];
      summaryEl.innerHTML = items.map(([k, v]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
    }

    function renderRows(payload) {
      rowsEl.innerHTML = (payload.runs || []).map((run) => {
        const total = Math.max(1, Number(run.total || 0));
        const progress = `${run.completed}/${total}`;
        const healthText = run.health_reason ? `${run.health}: ${run.health_reason}` : run.health;
        const warnings = (run.status_errors || []).map((item) => `<div>${esc(item)}</div>`).join("");
        return `
          <tr>
            <td>
              <div class="mono">${esc(run.run_tag)}</div>
              <div class="muted small mono">${esc(run.run_dir)}</div>
            </td>
            <td>
              ${pill(healthText, `health-${run.health}`)}
              ${pill(run.status_source, `src-${run.status_source}`)}
            </td>
            <td class="small">
              <div>${esc(progress)} done</div>
              <div class="muted">failed=${esc(run.failed)} partial=${esc(run.partial)} running=${esc(run.running)} pending=${esc(run.pending)}</div>
            </td>
            <td class="small nowrap">
              <div>${esc(run.last_event_age)}</div>
            </td>
            <td class="small">${stagePills(run) || '<span class="muted">-</span>'}</td>
            <td class="small">
              <div>${esc(run.config || "-")}</div>
              <div class="muted">${esc((run.requested_stage_tokens || []).join(",") || "-")}</div>
            </td>
            <td class="small mono">${esc(run.active_manifest_path || "-")}</td>
            <td class="small warnings">${warnings || '<span class="muted">-</span>'}</td>
          </tr>
        `;
      }).join("");
    }

    async function refresh() {
      const params = updateUrl();
      const response = await fetch(`/api/runs?${params.toString()}`, { cache: "no-store" });
      let payload = await response.json();
      if (Array.isArray(payload)) {
        payload = {
          log_root: "(legacy api)",
          updated_at: new Date().toLocaleString(),
          refresh_seconds: Number(document.getElementById("refresh_seconds").value || 5),
          runs: payload,
        };
      }
      if (payload && payload.error) {
        throw new Error(payload.error);
      }
      metaEl.textContent = `root=${payload.log_root} | updated=${payload.updated_at} | refresh=${payload.refresh_seconds}s`;
      renderSummary(payload);
      renderRows(payload);
      const refreshMs = Math.max(1, Number(payload.refresh_seconds || 5)) * 1000;
      clearTimeout(timer);
      timer = setTimeout(refresh, refreshMs);
    }

    for (const id of ids) {
      document.getElementById(id).addEventListener("change", () => {
        clearTimeout(timer);
        refresh();
      });
    }

    restoreFromUrl();
    refresh().catch((error) => {
      summaryEl.innerHTML = "";
      rowsEl.innerHTML = `<tr><td colspan="8" class="small warnings">load failed: ${esc(error && error.message ? error.message : error)}</td></tr>`;
      metaEl.textContent = `load failed: ${error && error.message ? error.message : error}`;
    });
  </script>
</body>
</html>
"""


def discover_access_urls(bind_host: str, port: int) -> list[str]:
    urls = []
    seen = set()

    def add_url(host: str) -> None:
        if ":" in host and not host.startswith("["):
            url = f"http://[{host}]:{port}"
        else:
            url = f"http://{host}:{port}"
        if url not in seen:
            seen.add(url)
            urls.append(url)

    if bind_host in ("127.0.0.1", "localhost"):
        add_url("127.0.0.1")
        return urls

    if bind_host in ("0.0.0.0", "::"):
        add_url("127.0.0.1")
        hostname = socket.gethostname()
        try:
            for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
                host = sockaddr[0]
                if family not in (socket.AF_INET, socket.AF_INET6):
                    continue
                if host.startswith("127.") or host == "::1":
                    continue
                add_url(host)
        except OSError:
            pass
        return urls

    add_url(bind_host)
    return urls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a shared run monitor dashboard over HTTP")
    parser.add_argument("--log_root", required=True, help="Shared dataset_pipeline_logs root")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--run_tags", default=None, help="Optional comma-separated run tags to pin")
    parser.add_argument("--pattern", default="20*", help="Default glob pattern under log_root")
    parser.add_argument("--limit", type=int, default=24, help="Default number of runs to show")
    parser.add_argument("--stall_seconds", type=int, default=1800, help="Mark active runs as stalled if no events arrive for this long")
    parser.add_argument("--refresh_seconds", type=int, default=5, help="Browser polling interval")
    return parser


class MonitorApp:
    def __init__(self, args: argparse.Namespace):
        self.log_root = Path(args.log_root).resolve()
        self.default_run_tags = [item.strip() for item in str(args.run_tags or "").split(",") if item.strip()] or None
        self.default_pattern = args.pattern
        self.default_limit = args.limit
        self.default_stall_seconds = args.stall_seconds
        self.default_refresh_seconds = args.refresh_seconds

    def payload(self, query: dict[str, list[str]]) -> dict:
        run_tags_raw = (query.get("run_tags") or [""])[0].strip()
        run_tags = [item.strip() for item in run_tags_raw.split(",") if item.strip()] if run_tags_raw else self.default_run_tags
        pattern = (query.get("pattern") or [self.default_pattern])[0]
        limit = int((query.get("limit") or [self.default_limit])[0])
        stall_seconds = int((query.get("stall_seconds") or [self.default_stall_seconds])[0])
        refresh_seconds = int((query.get("refresh_seconds") or [self.default_refresh_seconds])[0])
        runs = summarize_runs(
            self.log_root,
            run_tags=run_tags,
            pattern=pattern,
            limit=limit,
            stall_seconds=stall_seconds,
        )
        return {
            "log_root": str(self.log_root),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "refresh_seconds": refresh_seconds,
            "pattern": pattern,
            "run_tags": run_tags or [],
            "limit": limit,
            "stall_seconds": stall_seconds,
            "runs": runs,
        }


def make_handler(app: MonitorApp):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_html(self, html: str, status=200):
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send_html(HTML_PAGE)
                return
            if parsed.path == "/api/runs":
                try:
                    payload = app.payload(query)
                except Exception as error:
                    self._send_json({"error": str(error)}, status=500)
                    return
                self._send_json(payload)
                return
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self._send_json({"error": f"Unsupported path: {parsed.path}"}, status=404)

    return Handler


def main() -> int:
    args = build_parser().parse_args()
    if not Path(args.log_root).is_dir():
        raise SystemExit(f"log_root not found: {args.log_root}")

    app = MonitorApp(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Server bind: {args.host}:{args.port}")
    print(f"Shared log root: {app.log_root}")
    for url in discover_access_urls(args.host, args.port):
        print(f"Dashboard URL: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
