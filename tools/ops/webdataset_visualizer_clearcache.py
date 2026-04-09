#!/usr/bin/env python3
"""Local web viewer for inspecting WebDataset shards.

Deprecated:
    Prefer ``tools/ops/rerun_webdataset_visualizer.py`` for the default
    shard/episode inspection workflow. This web viewer remains as a fallback
    for environments where Rerun is unavailable.

Example:
    python tools/ops/webdataset_visualizer.py \
        --input /path/to/shards_or_single_tar \
        --host 127.0.0.1 \
        --port 8765
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import socket
import sys
import tarfile
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.mano_codec import (
    DEFAULT_MANO_DIR,
    MANO_CENTER_IDX,
    MANO_FLAT_HAND_MEAN,
    MANO_PCA_DIMS,
    MANO_SCHEMA,
    build_manopth_models,
    decode_mano_sample_array,
    rot6d_to_axis_angle,
    run_manopth_mano,
)

SAMPLE_MEMBER_SUFFIXES = (
    (".image.jpg", "image_bytes"),
    (".lowdim.npy", "lowdim_bytes"),
    (".mano.npy", "mano_bytes"),
    (".meta.json", "meta_bytes"),
)
REQUIRED_SAMPLE_FIELDS = ("image_bytes", "lowdim_bytes", "meta_bytes")

LOWDIM_SEGMENTS = (
    {
        "name": "left_wrist_world",
        "start": 0,
        "end": 3,
        "description": "Left wrist joint position in world coordinates.",
    },
    {
        "name": "right_wrist_world",
        "start": 3,
        "end": 6,
        "description": "Right wrist joint position in world coordinates.",
    },
    {
        "name": "left_root_rot6d",
        "start": 6,
        "end": 12,
        "description": "Left MANO root orientation in rot6d, expressed in world coordinates.",
    },
    {
        "name": "right_root_rot6d",
        "start": 12,
        "end": 18,
        "description": "Right MANO root orientation in rot6d, expressed in world coordinates.",
    },
    {
        "name": "left_fingertips_world",
        "start": 18,
        "end": 33,
        "description": "Left fingertip 3D points from MANO joints, shape (5, 3), world coordinates.",
    },
    {
        "name": "right_fingertips_world",
        "start": 33,
        "end": 48,
        "description": "Right fingertip 3D points from MANO joints, shape (5, 3), world coordinates.",
    },
    {
        "name": "next_left_wrist_world",
        "start": 48,
        "end": 51,
        "description": "Next-frame left wrist joint position in world coordinates.",
    },
    {
        "name": "next_right_wrist_world",
        "start": 51,
        "end": 54,
        "description": "Next-frame right wrist joint position in world coordinates.",
    },
    {
        "name": "next_left_root_rot6d",
        "start": 54,
        "end": 60,
        "description": "Next-frame left root orientation in rot6d.",
    },
    {
        "name": "next_right_root_rot6d",
        "start": 60,
        "end": 66,
        "description": "Next-frame right root orientation in rot6d.",
    },
    {
        "name": "next_left_fingertips_world",
        "start": 66,
        "end": 81,
        "description": "Next-frame left fingertips, shape (5, 3), world coordinates.",
    },
    {
        "name": "next_right_fingertips_world",
        "start": 81,
        "end": 96,
        "description": "Next-frame right fingertips, shape (5, 3), world coordinates.",
    },
    {
        "name": "camera_w2c",
        "start": 96,
        "end": 112,
        "description": "Camera extrinsic as a 4x4 world-to-camera matrix, flattened row-major.",
    },
    {
        "name": "camera_intrinsic",
        "start": 112,
        "end": 116,
        "description": "Pinhole intrinsic [fx, fy, cx, cy].",
    },
)

FRAME_SUFFIX_RE = re.compile(r"_f\d+$")
FRAME_INDEX_RE = re.compile(r"_f(\d+)$")
MANO_JOINT_TREE = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]
FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)
DEMO_RX_3X3 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WebDataset Visualizer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f2e8;
      --panel: #fffdf7;
      --line: #d7d0be;
      --ink: #1e1b16;
      --muted: #6c6558;
      --accent: #05668d;
      --danger: #b42318;
      --warn: #c47f00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iosevka Aile", "IBM Plex Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(5, 102, 141, 0.10), transparent 28%),
        radial-gradient(circle at top left, rgba(196, 127, 0, 0.08), transparent 25%),
        var(--bg);
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 12px;
      padding: 18px;
    }
    .bar, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(30, 27, 22, 0.05);
    }
    .bar {
      padding: 14px 16px;
    }
    .title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0.02em;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .controls .wide {
      grid-column: span 2;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    input, select, button, textarea {
      font: inherit;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 10px;
      padding: 10px 12px;
      min-height: 42px;
    }
    button {
      border: 0;
      background: var(--ink);
      color: white;
      border-radius: 10px;
      padding: 10px 14px;
      min-height: 42px;
      cursor: pointer;
    }
    button.secondary {
      background: #ddd5c3;
      color: var(--ink);
    }
    button:disabled {
      opacity: 0.45;
      cursor: default;
    }
    .main {
      display: grid;
      grid-template-columns: minmax(420px, 1.3fr) minmax(320px, 0.9fr);
      gap: 12px;
      min-height: 0;
    }
    .viewer {
      padding: 14px;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 12px;
      min-height: 0;
    }
    .meta {
      padding: 14px;
      display: grid;
      grid-template-rows: auto auto auto 1fr;
      gap: 12px;
      min-height: 0;
    }
    .nav {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .nav .grow {
      flex: 1 1 220px;
      min-width: 180px;
    }
    .nav .episode-select {
      min-width: 220px;
      max-width: 420px;
    }
    .counter {
      font-size: 13px;
      color: var(--muted);
      padding: 0 8px;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
      min-height: auto;
      padding: 0;
      border: 0;
      background: transparent;
    }
    .statusline {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      border: 1px solid transparent;
    }
    .ok {
      background: rgba(5, 102, 141, 0.08);
      color: var(--accent);
      border-color: rgba(5, 102, 141, 0.18);
    }
    .broken {
      background: rgba(180, 35, 24, 0.08);
      color: var(--danger);
      border-color: rgba(180, 35, 24, 0.16);
    }
    .warn {
      background: rgba(196, 127, 0, 0.10);
      color: var(--warn);
      border-color: rgba(196, 127, 0, 0.18);
    }
    .image-wrap {
      min-height: 280px;
      height: 100%;
      border: 1px dashed var(--line);
      border-radius: 14px;
      overflow: auto;
      background:
        linear-gradient(135deg, rgba(5, 102, 141, 0.04), rgba(196, 127, 0, 0.04)),
        white;
      display: grid;
      place-items: center;
      padding: 12px;
    }
    .image-wrap img {
      max-width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.10);
    }
    .empty {
      color: var(--muted);
      font-size: 14px;
    }
    .kv {
      display: grid;
      gap: 8px;
    }
    .kv-row {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      padding: 10px 12px;
    }
    .kv-key {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }
    .mono, pre {
      font-family: "Iosevka", "SFMono-Regular", monospace;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f9f7f1;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      overflow: auto;
      min-height: 0;
    }
    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .segments table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .segments th, .segments td {
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
    }
    .segments th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    @media (max-width: 1100px) {
      .controls {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .controls .wide {
        grid-column: span 2;
      }
      .main {
        grid-template-columns: 1fr;
      }
      .split {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="bar">
      <div class="title">
        <div>
          <h1>WebDataset Visualizer</h1>
          <div class="hint" id="datasetHint">Loading shard index...</div>
        </div>
        <div class="hint mono" id="serverHint"></div>
      </div>
      <div class="controls">
        <label class="wide">
          Search Key / Clip / Instruction
          <input id="searchInput" type="text" placeholder="substring match">
        </label>
        <label>
          Presence
          <select id="presenceSelect">
            <option value="">All</option>
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
          </select>
        </label>
        <label>
          Render Mode
          <select id="renderModeSelect"></select>
        </label>
        <label class="wide">
          Shard
          <select id="shardSelect"></select>
        </label>
        <label>
          Jump To
          <input id="jumpInput" type="number" min="1" step="1" placeholder="1-based">
        </label>
        <label>
          Apply
          <button id="applyBtn" class="secondary">Filter</button>
        </label>
      </div>
    </div>

    <div class="bar nav">
      <button id="prevEpisodeBtn">Prev Episode</button>
      <button id="nextEpisodeBtn">Next Episode</button>
      <label class="episode-select">
        Episode
        <select id="episodeSelect"></select>
      </label>
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <button id="playBtn">Play</button>
      <label>
        FPS
        <select id="fpsSelect">
          <option value="1">1</option>
          <option value="2">2</option>
          <option value="4">4</option>
          <option value="12">12</option>
          <option value="24">24</option>
          <option value="30" selected>30</option>
        </select>
      </label>
      <label class="grow">
        Frame Scrubber
        <input id="scrubInput" type="range" min="1" max="1" value="1">
      </label>
      <span class="counter" id="counter">0 / 0</span>
      <span class="hint mono" id="episodeCounter"></span>
      <span class="hint mono" id="sampleKey"></span>
    </div>

    <div class="main">
      <section class="panel viewer">
        <div class="statusline">
          <span id="healthBadge" class="badge ok">OK</span>
          <span id="shardName"></span>
          <span id="clipId"></span>
        </div>
        <div class="statusline mono" id="sampleSummary"></div>
        <div class="image-wrap" id="imageWrap">
          <div class="empty">No sample loaded.</div>
        </div>
      </section>

      <aside class="panel meta">
        <div class="split">
          <div class="kv">
            <div class="kv-row">
              <div class="kv-key">Instruction</div>
              <div id="instructionText">-</div>
            </div>
            <div class="kv-row">
              <div class="kv-key">Meta Summary</div>
              <div id="metaSummary">-</div>
            </div>
          </div>
          <div class="kv">
            <div class="kv-row">
              <div class="kv-key">Lowdim Summary</div>
              <div id="lowdimSummary">-</div>
            </div>
            <div class="kv-row">
              <div class="kv-key">Render Notes</div>
              <div id="renderNotes">-</div>
            </div>
            <div class="kv-row">
              <div class="kv-key">Errors</div>
              <div id="errorSummary">None</div>
            </div>
          </div>
        </div>

        <div class="segments kv-row">
          <div class="kv-key">Lowdim Layout</div>
          <div id="segmentsWrap">-</div>
        </div>

        <div class="split" style="min-height: 0;">
          <div style="min-height: 0; display: grid;">
            <div class="kv-key">Meta JSON</div>
            <pre id="metaPre">-</pre>
          </div>
          <div style="min-height: 0; display: grid;">
            <div class="kv-key">Lowdim Preview</div>
            <pre id="lowdimPre">-</pre>
          </div>
        </div>
      </aside>
    </div>
  </div>

  <script>
    const state = {
      config: null,
      cacheScope: "boot",
      samples: [],
      filtered: [],
      currentPosition: 0,
      playTimer: null,
      isPlaying: false,
      loadingSample: false,
      loadToken: 0,
      samplePayloadCache: new Map(),
      imageElement: null,
    };

    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }
      return await res.json();
    }

    function el(id) {
      return document.getElementById(id);
    }

    function sampleCacheKey(sampleId, renderMode, warmImage = true) {
      return `${state.cacheScope}:${sampleId}:${renderMode}:${warmImage ? "warm" : "cold"}`;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function buildShardOptions() {
      const select = el("shardSelect");
      const shardNames = Array.from(new Set(state.samples.map((item) => item.shard_name))).sort();
      select.innerHTML = `<option value="">All Shards</option>` +
        shardNames.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
    }

    function buildRenderModeOptions() {
      const select = el("renderModeSelect");
      const modes = state.config.available_render_modes || ["keypoint", "mano"];
      select.innerHTML = modes.map((mode) => `<option value="${escapeHtml(mode)}">${escapeHtml(mode)}</option>`).join("");
      select.value = state.config.default_render_mode || "keypoint";
    }

    function updateEpisodeOptions() {
      const select = el("episodeSelect");
      const positions = getEpisodeStartPositions();
      const currentEpisodeKey = getCurrentEpisodeKey();
      if (positions.length === 0) {
        select.innerHTML = `<option value="">No Episodes</option>`;
        select.disabled = true;
        return;
      }
      select.disabled = false;
      select.innerHTML = positions.map((item, idx) => `
        <option value="${escapeHtml(item.episode_key)}">
          ${escapeHtml(`${idx + 1}. ${item.episode_key}`)}
        </option>
      `).join("");
      if (currentEpisodeKey) {
        select.value = currentEpisodeKey;
      }
    }

    function applyFilters() {
      const query = el("searchInput").value.trim().toLowerCase();
      const presence = el("presenceSelect").value;
      const shard = el("shardSelect").value;
      state.filtered = state.samples.filter((item) => {
        if (presence !== "" && String(item.presence ?? "") !== presence) {
          return false;
        }
        if (shard && item.shard_name !== shard) {
          return false;
        }
        if (!query) {
          return true;
        }
        const haystack = [
          item.key,
          item.clip_id || "",
          item.instruction_preview || "",
          item.shard_name,
        ].join(" ").toLowerCase();
        return haystack.includes(query);
      });

      const jump = Number(el("jumpInput").value || "1");
      let nextPos = Number.isFinite(jump) ? Math.max(0, jump - 1) : 0;
      if (nextPos >= state.filtered.length) {
        nextPos = Math.max(0, state.filtered.length - 1);
      }
      state.currentPosition = nextPos;
      updateEpisodeOptions();
      renderCounter();
      if (state.filtered.length > 0) {
        loadCurrentSample();
      } else {
        clearSample("No filtered samples.");
      }
    }

    function renderCounter() {
      const positions = getEpisodeStartPositions();
      const currentEpisodeIdx = getCurrentEpisodeIndex();
      const currentEpisodeKey = getCurrentEpisodeKey();
      const episodeBounds = getCurrentEpisodeBounds();
      const total = state.filtered.length;
      const episodeSize = episodeBounds ? episodeBounds.endExclusive - episodeBounds.start : 0;
      const episodeFrame =
        episodeBounds ? state.currentPosition - episodeBounds.start + 1 : 0;
      el("counter").textContent = `${episodeFrame} / ${episodeSize}`;
      el("episodeCounter").textContent =
        positions.length === 0
          ? ""
          : `episode ${currentEpisodeIdx + 1} / ${positions.length}: ${currentEpisodeKey}`;
      if (currentEpisodeKey && !el("episodeSelect").disabled) {
        el("episodeSelect").value = currentEpisodeKey;
      }
      el("prevEpisodeBtn").disabled = positions.length === 0 || currentEpisodeIdx <= 0;
      el("nextEpisodeBtn").disabled = positions.length === 0 || currentEpisodeIdx >= positions.length - 1;
      el("prevBtn").disabled = total === 0 || state.currentPosition <= 0;
      el("nextBtn").disabled = total === 0;
      el("scrubInput").min = "1";
      el("scrubInput").max = String(Math.max(episodeSize, 1));
      el("scrubInput").value = String(Math.max(episodeFrame, 1));
      el("playBtn").disabled = total === 0;
    }

    function getCurrentEpisodeKey() {
      if (state.filtered.length === 0) {
        return null;
      }
      return state.filtered[state.currentPosition].episode_key;
    }

    function getEpisodeStartPositions() {
      const positions = [];
      let prevEpisodeKey = null;
      state.filtered.forEach((item, idx) => {
        if (item.episode_key !== prevEpisodeKey) {
          positions.push({ episode_key: item.episode_key, start: idx });
          prevEpisodeKey = item.episode_key;
        }
      });
      return positions;
    }

    function getCurrentEpisodeIndex() {
      const currentEpisodeKey = getCurrentEpisodeKey();
      const positions = getEpisodeStartPositions();
      if (!currentEpisodeKey) {
        return -1;
      }
      return positions.findIndex((item) => item.episode_key === currentEpisodeKey);
    }

    function getCurrentEpisodeBounds() {
      const positions = getEpisodeStartPositions();
      const currentEpisodeIdx = getCurrentEpisodeIndex();
      if (currentEpisodeIdx < 0 || currentEpisodeIdx >= positions.length) {
        return null;
      }
      const start = positions[currentEpisodeIdx].start;
      const endExclusive =
        currentEpisodeIdx + 1 < positions.length
          ? positions[currentEpisodeIdx + 1].start
          : state.filtered.length;
      return { start, endExclusive };
    }

    async function jumpEpisode(delta) {
      stopPlayback();
      const positions = getEpisodeStartPositions();
      const currentEpisodeIdx = getCurrentEpisodeIndex();
      if (positions.length === 0 || currentEpisodeIdx < 0) {
        return;
      }
      const targetIdx = currentEpisodeIdx + delta;
      if (targetIdx < 0 || targetIdx >= positions.length) {
        return;
      }
      state.currentPosition = positions[targetIdx].start;
      await loadCurrentSample();
    }

    async function jumpToEpisodeKey(episodeKey) {
      stopPlayback();
      const positions = getEpisodeStartPositions();
      const target = positions.find((item) => item.episode_key === episodeKey);
      if (!target) {
        return;
      }
      state.currentPosition = target.start;
      await loadCurrentSample();
    }

    function stopPlayback() {
      if (state.playTimer !== null) {
        clearInterval(state.playTimer);
        state.playTimer = null;
      }
      state.isPlaying = false;
      el("playBtn").textContent = "Play";
    }

    function startPlayback() {
      stopPlayback();
      if (state.filtered.length === 0) {
        return;
      }
      const playbackEpisodeKey = getCurrentEpisodeKey();
      const fps = Math.max(1, Number(el("fpsSelect").value || "30"));
      const intervalMs = Math.max(1, Math.round(1000 / fps));
      state.isPlaying = true;
      el("playBtn").textContent = "Pause";
      state.playTimer = setInterval(async () => {
        if (state.loadingSample || state.filtered.length === 0) {
          return;
        }
        const currentEpisodeKey = getCurrentEpisodeKey();
        if (!playbackEpisodeKey || currentEpisodeKey !== playbackEpisodeKey) {
          stopPlayback();
          return;
        }
        const positions = getEpisodeStartPositions();
        const currentEpisodeIdx = getCurrentEpisodeIndex();
        if (currentEpisodeIdx < 0 || currentEpisodeIdx >= positions.length) {
          stopPlayback();
          return;
        }
        const episodeBounds = getCurrentEpisodeBounds();
        if (!episodeBounds) {
          stopPlayback();
          return;
        }
        const { endExclusive } = episodeBounds;
        if (state.currentPosition + 1 >= endExclusive) {
          stopPlayback();
          return;
        } else {
          state.currentPosition += 1;
        }
        await loadCurrentSample();
      }, intervalMs);
    }

    function togglePlayback() {
      if (state.isPlaying) {
        stopPlayback();
      } else {
        startPlayback();
      }
    }

    function clearSample(message) {
      el("sampleKey").textContent = "";
      el("shardName").textContent = "";
      el("clipId").textContent = "";
      el("sampleSummary").textContent = "";
      el("instructionText").textContent = "-";
      el("metaSummary").textContent = "-";
      el("lowdimSummary").textContent = "-";
      el("renderNotes").textContent = "-";
      el("errorSummary").textContent = message || "None";
      el("metaPre").textContent = "-";
      el("lowdimPre").textContent = "-";
      el("segmentsWrap").innerHTML = "-";
      el("healthBadge").className = "badge warn";
      el("healthBadge").textContent = "EMPTY";
      state.imageElement = null;
      el("imageWrap").innerHTML = `<div class="empty">${escapeHtml(message || "No sample loaded.")}</div>`;
    }

    function ensureImageElement() {
      if (state.imageElement) {
        return state.imageElement;
      }
      const wrap = el("imageWrap");
      wrap.innerHTML = "";
      const image = document.createElement("img");
      image.alt = "frame";
      wrap.appendChild(image);
      state.imageElement = image;
      return image;
    }

    function preloadPayload(sampleId, renderMode, options = {}) {
      const warmImage = options.warmImage !== false;
      const cacheKey = sampleCacheKey(sampleId, renderMode, warmImage);
      if (state.samplePayloadCache.has(cacheKey)) {
        return state.samplePayloadCache.get(cacheKey);
      }
      const promise = fetchJson(
        `/api/sample/${sampleId}?render_mode=${encodeURIComponent(renderMode)}&warm_image=${warmImage ? "1" : "0"}`
      )
        .catch((error) => {
          state.samplePayloadCache.delete(cacheKey);
          throw error;
        });
      state.samplePayloadCache.set(cacheKey, promise);
      return promise;
    }

    function prefetchEpisodeNeighborhood(renderMode) {
      if (state.filtered.length === 0) {
        return;
      }
      const positions = getEpisodeStartPositions();
      const currentEpisodeIdx = getCurrentEpisodeIndex();
      if (currentEpisodeIdx < 0 || currentEpisodeIdx >= positions.length) {
        return;
      }
      const start = positions[currentEpisodeIdx].start;
      const endExclusive =
        currentEpisodeIdx + 1 < positions.length
          ? positions[currentEpisodeIdx + 1].start
          : state.filtered.length;
      const prefetchCount = renderMode === "mano" ? 2 : 6;
      const maxOffset = Math.min(prefetchCount + 1, endExclusive - start);
      for (let offset = 1; offset < maxOffset; offset += 1) {
        const idx = start + ((state.currentPosition - start + offset) % Math.max(1, endExclusive - start));
        const item = state.filtered[idx];
        void preloadPayload(item.id, renderMode, { warmImage: false });
      }
    }

    function renderSampleData(data) {
      const broken = (data.errors || []).length > 0 || data.broken;

      el("healthBadge").className = `badge ${broken ? "broken" : "ok"}`;
      el("healthBadge").textContent = broken ? "BROKEN" : "OK";
      el("shardName").textContent = data.shard_name;
      el("clipId").textContent = data.clip_id ? `clip_id=${data.clip_id}` : "clip_id=<missing>";
      el("sampleSummary").textContent = `${data.key} | shard=${data.shard_name} | presence=${data.presence ?? "?"} | render=${data.render_mode}`;
      el("instructionText").textContent = data.instruction_display || "-";
      const manoInfo = data.mano || {};
      const manoStatus = manoInfo.schema || manoInfo.error || "absent";
      el("metaSummary").textContent = `instruction_num=${data.instruction_num ?? "?"} | presence=${data.presence ?? "?"} | mano=${manoStatus}`;

      const lowdimInfo = data.lowdim || {};
      if (lowdimInfo.error) {
        el("lowdimSummary").textContent = `error: ${lowdimInfo.error}`;
      } else {
        el("lowdimSummary").textContent =
          `shape=${lowdimInfo.shape} | dtype=${lowdimInfo.dtype} | coord=${lowdimInfo.coordinate_system || "world"} | min=${lowdimInfo.min} | max=${lowdimInfo.max} | mean=${lowdimInfo.mean}`;
      }

      el("renderNotes").textContent =
        (data.render_notes && data.render_notes.length > 0) ? data.render_notes.join(" | ") : "-";
      el("errorSummary").textContent = (data.errors && data.errors.length > 0) ? data.errors.join(" | ") : "None";
      el("metaPre").textContent = data.meta_pretty || "-";
      el("lowdimPre").textContent = data.lowdim_preview || "-";
      renderSegments(data.lowdim_segments || []);

      if (data.image_url) {
        const image = ensureImageElement();
        image.alt = data.key;
        image.src = data.image_url;
      } else {
        state.imageElement = null;
        el("imageWrap").innerHTML = `<div class="empty">No image payload.</div>`;
      }
    }

    function renderSegments(segments) {
      if (!segments || segments.length === 0) {
        el("segmentsWrap").textContent = "-";
        return;
      }
      const rows = segments.map((seg) => `
        <tr>
          <td class="mono">${escapeHtml(seg.name)}</td>
          <td class="mono">${seg.start}:${seg.end}</td>
          <td class="mono">${escapeHtml(seg.shape)}</td>
          <td>${escapeHtml(seg.description || "-")}</td>
          <td class="mono">${escapeHtml(seg.preview)}</td>
        </tr>
      `).join("");
      el("segmentsWrap").innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Segment</th>
              <th>Slice</th>
              <th>Shape</th>
              <th>Description</th>
              <th>Preview</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }

    async function loadCurrentSample() {
      if (state.filtered.length === 0) {
        clearSample("No filtered samples.");
        return;
      }
      const loadToken = ++state.loadToken;
      state.loadingSample = true;
      try {
        renderCounter();
        const summary = state.filtered[state.currentPosition];
        el("sampleKey").textContent = summary.key;
        el("sampleSummary").textContent = "Loading sample detail...";
        const renderMode = el("renderModeSelect").value;
        const data = await preloadPayload(summary.id, renderMode, { warmImage: true });
        if (loadToken !== state.loadToken) {
          return;
        }
        renderSampleData(data);
        setTimeout(() => prefetchEpisodeNeighborhood(renderMode), 0);
      } finally {
        state.loadingSample = false;
      }
    }

    async function init() {
      state.config = await fetchJson("/api/config");
      state.cacheScope = state.config.cache_token || String(Date.now());
      state.samplePayloadCache.clear();
      state.samples = await fetchJson("/api/index");
      buildShardOptions();
      buildRenderModeOptions();
      el("datasetHint").textContent =
        `${state.config.tar_count} tar(s), ${state.samples.length} sample(s), input=${state.config.input_path}`;
      el("serverHint").textContent = `${state.config.host}:${state.config.port}`;
      state.currentPosition = Math.min(Math.max(state.config.start_index, 0), Math.max(state.samples.length - 1, 0));
      applyFilters();
    }

    el("applyBtn").addEventListener("click", () => {
      stopPlayback();
      applyFilters();
    });
    el("prevEpisodeBtn").addEventListener("click", async () => {
      await jumpEpisode(-1);
    });
    el("nextEpisodeBtn").addEventListener("click", async () => {
      await jumpEpisode(1);
    });
    el("episodeSelect").addEventListener("change", async () => {
      await jumpToEpisodeKey(el("episodeSelect").value);
    });
    el("prevBtn").addEventListener("click", async () => {
      stopPlayback();
      if (state.filtered.length === 0) return;
      if (state.currentPosition <= 0) return;
      state.currentPosition -= 1;
      await loadCurrentSample();
    });
    el("nextBtn").addEventListener("click", async () => {
      stopPlayback();
      if (state.filtered.length === 0) return;
      state.currentPosition = (state.currentPosition + 1) % state.filtered.length;
      await loadCurrentSample();
    });
    el("playBtn").addEventListener("click", togglePlayback);
    el("fpsSelect").addEventListener("change", () => {
      if (state.isPlaying) {
        startPlayback();
      }
    });
    el("scrubInput").addEventListener("input", async () => {
      stopPlayback();
      const episodeBounds = getCurrentEpisodeBounds();
      if (!episodeBounds) {
        return;
      }
      const nextEpisodeOffset = Math.max(0, Number(el("scrubInput").value || "1") - 1);
      const nextPos = Math.min(
        episodeBounds.start + nextEpisodeOffset,
        Math.max(episodeBounds.start, episodeBounds.endExclusive - 1),
      );
      if (nextPos === state.currentPosition) {
        return;
      }
      state.currentPosition = nextPos;
      await loadCurrentSample();
    });
    el("searchInput").addEventListener("keydown", (event) => {
      if (event.key === "Enter") applyFilters();
    });
    el("renderModeSelect").addEventListener("change", async () => {
      stopPlayback();
      await loadCurrentSample();
    });
    el("jumpInput").addEventListener("keydown", (event) => {
      if (event.key === "Enter") applyFilters();
    });
    window.addEventListener("keydown", async (event) => {
      if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        stopPlayback();
        if (state.currentPosition > 0) {
          state.currentPosition -= 1;
          await loadCurrentSample();
        }
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        stopPlayback();
        if (state.filtered.length > 0) {
          state.currentPosition = (state.currentPosition + 1) % state.filtered.length;
          await loadCurrentSample();
        }
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        await jumpEpisode(-1);
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        await jumpEpisode(1);
      }
      if (event.key === " ") {
        event.preventDefault();
        togglePlayback();
      }
    });

    init().catch((error) => {
      clearSample(`Failed to initialize viewer: ${error.message}`);
      el("datasetHint").textContent = "Initialization failed.";
    });
  </script>
</body>
</html>
"""


@dataclass
class SampleSummary:
    id: int
    key: str
    episode_key: str
    shard_path: str
    shard_name: str
    clip_id: Optional[str]
    instruction_preview: str
    instruction_num: Optional[int]
    presence: Optional[int]
    broken: bool
    missing_fields: list[str]


def parse_frame_index(sample_key: str) -> int:
    match = FRAME_INDEX_RE.search(sample_key)
    if not match:
        raise ValueError(f"Failed to parse frame index from sample key: {sample_key}")
    return int(match.group(1))


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "episode"


def split_sample_member_name(member_name: str):
    for suffix, field_name in SAMPLE_MEMBER_SUFFIXES:
        if member_name.endswith(suffix):
            return member_name[: -len(suffix)], suffix, field_name
    return None, None, None


def new_sample_record(sample_key: str):
    return {
        "key": sample_key,
        "image_bytes": None,
        "lowdim_bytes": None,
        "mano_bytes": None,
        "meta_bytes": None,
    }


def iter_shard_samples(shard_path: str) -> Iterable[dict]:
    current_sample = None
    with tarfile.open(shard_path, "r:*") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            sample_key, _, field_name = split_sample_member_name(member.name)
            if sample_key is None:
                continue

            member_file = tar_reader.extractfile(member)
            if member_file is None:
                continue
            member_bytes = member_file.read()

            if current_sample is None:
                current_sample = new_sample_record(sample_key)
            elif current_sample["key"] != sample_key:
                yield current_sample
                current_sample = new_sample_record(sample_key)

            current_sample[field_name] = member_bytes

    if current_sample is not None:
        yield current_sample


def decode_meta(meta_bytes: Optional[bytes]):
    if meta_bytes is None:
        return None, "missing meta.json"
    try:
        return json.loads(meta_bytes.decode("utf-8")), None
    except Exception as error:
        return None, str(error)


def normalize_instruction(meta: Optional[dict]) -> str:
    if not meta:
        return ""
    instruction = meta.get("instruction")
    if isinstance(instruction, list):
        return " | ".join(str(item) for item in instruction if str(item).strip())
    if instruction is None:
        return ""
    return str(instruction)


def truncate_text(text: str, max_len: int = 180) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def sample_key_to_episode_key(sample_key: str) -> str:
    return FRAME_SUFFIX_RE.sub("", sample_key)


def build_sample_summary(sample: dict, shard_path: str, sample_id: int) -> SampleSummary:
    meta, meta_error = decode_meta(sample["meta_bytes"])
    missing_fields = [field_name for field_name in REQUIRED_SAMPLE_FIELDS if sample.get(field_name) is None]
    instruction_text = truncate_text(normalize_instruction(meta))
    clip_id = None if meta is None else meta.get("clip_id")
    instruction_num = None if meta is None else meta.get("instruction_num")
    presence = None if meta is None else meta.get("presence")
    broken = bool(missing_fields) or meta_error is not None
    return SampleSummary(
        id=sample_id,
        key=sample["key"],
        episode_key=str(clip_id) if clip_id is not None else sample_key_to_episode_key(sample["key"]),
        shard_path=shard_path,
        shard_name=os.path.basename(shard_path),
        clip_id=str(clip_id) if clip_id is not None else None,
        instruction_preview=instruction_text,
        instruction_num=int(instruction_num) if isinstance(instruction_num, (int, np.integer)) else instruction_num,
        presence=int(presence) if isinstance(presence, (int, np.integer)) else presence,
        broken=broken,
        missing_fields=missing_fields,
    )


def resolve_tar_paths(input_path: str) -> list[str]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".tar":
            raise ValueError(f"Input file is not a .tar: {path}")
        return [str(path)]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    tar_paths = sorted(str(child.resolve()) for child in path.iterdir() if child.is_file() and child.suffix == ".tar")
    if not tar_paths:
        raise RuntimeError(f"No .tar files found under {path}")
    return tar_paths


def scan_samples(
    tar_paths: list[str],
    *,
    sample_limit: Optional[int],
    episode_limit: Optional[int],
    filter_key: str,
    filter_presence: Optional[int],
) -> list[SampleSummary]:
    entries: list[SampleSummary] = []
    filter_key_lower = filter_key.lower()
    matched_in_shard = 0
    selected_episodes: list[str] = []
    selected_episode_set: set[str] = set()

    print(f"Scanning {len(tar_paths)} tar shard(s) for samples...", flush=True)

    for shard_idx, shard_path in enumerate(tar_paths, start=1):
        matched_in_shard = 0
        print(
            f"[scan] {shard_idx}/{len(tar_paths)} {os.path.basename(shard_path)}",
            flush=True,
        )
        for sample in iter_shard_samples(shard_path):
            summary = build_sample_summary(sample, shard_path, len(entries))
            if filter_key_lower:
                haystack = " ".join(
                    [
                        summary.key,
                        summary.episode_key,
                        summary.clip_id or "",
                        summary.instruction_preview or "",
                        summary.shard_name,
                    ]
                ).lower()
                if filter_key_lower not in haystack:
                    continue
            if filter_presence is not None and summary.presence != filter_presence:
                continue

            if episode_limit is not None:
                if summary.episode_key not in selected_episode_set:
                    if len(selected_episode_set) >= episode_limit:
                        print(
                            f"Reached episode limit {episode_limit}; stopping scan.",
                            flush=True,
                        )
                        return entries
                    selected_episode_set.add(summary.episode_key)
                    selected_episodes.append(summary.episode_key)
                    print(
                        f"  selected_episode[{len(selected_episodes)}]={summary.episode_key}",
                        flush=True,
                    )

            entries.append(summary)
            matched_in_shard += 1
            if len(entries) <= 5 or len(entries) % 500 == 0:
                print(
                    f"  matched={len(entries)} current_shard={matched_in_shard}",
                    flush=True,
                )
            if sample_limit is not None and len(entries) >= sample_limit:
                print(
                    f"Reached sample limit {sample_limit}; stopping scan.",
                    flush=True,
                )
                return entries
        print(
            f"  done shard {shard_idx}/{len(tar_paths)} matched_in_shard={matched_in_shard} total={len(entries)}",
            flush=True,
        )
    if selected_episodes:
        print(f"Selected episode(s): {', '.join(selected_episodes)}", flush=True)
    print(f"Finished scan: {len(entries)} matched sample(s).", flush=True)
    return entries

def build_lowdim_segments(lowdim_array: np.ndarray) -> list[dict]:
    if lowdim_array.ndim == 0:
        return []
    flat = lowdim_array.reshape(-1)
    if flat.shape[0] != 116:
        preview = ", ".join(f"{value:.4f}" for value in flat[: min(8, flat.shape[0])].tolist())
        return [
            {
                "name": "all",
                "start": 0,
                "end": int(flat.shape[0]),
                "shape": str(tuple(lowdim_array.shape)),
                "preview": preview,
            }
        ]

    segments = []
    for item in LOWDIM_SEGMENTS:
        segment = flat[item["start"] : item["end"]]
        segments.append(
            {
                "name": item["name"],
                "start": item["start"],
                "end": item["end"],
                "shape": str(tuple(segment.shape)),
                "description": item["description"],
                "preview": ", ".join(f"{value:.4f}" for value in segment[: min(6, len(segment))].tolist()),
            }
        )
    return segments


def summarize_lowdim(lowdim_bytes: Optional[bytes]):
    if lowdim_bytes is None:
        return {
            "error": "missing lowdim.npy",
        }
    try:
        array = np.load(io.BytesIO(lowdim_bytes), allow_pickle=False)
    except Exception as error:
        return {
            "error": str(error),
        }

    flat = array.reshape(-1).astype(np.float32) if array.size > 0 else np.array([], dtype=np.float32)
    preview = "[]"
    if flat.size > 0:
        preview_values = [round(float(value), 6) for value in flat[: min(32, flat.size)].tolist()]
        preview = json.dumps(preview_values, ensure_ascii=False, indent=2)

    return {
        "shape": str(tuple(array.shape)),
        "dtype": str(array.dtype),
        "coordinate_system": "world",
        "min": round(float(array.min()), 6) if array.size > 0 else None,
        "max": round(float(array.max()), 6) if array.size > 0 else None,
        "mean": round(float(array.mean()), 6) if array.size > 0 else None,
        "preview": preview,
        "segments": build_lowdim_segments(array),
        "array": array,
    }


def summarize_mano(mano_bytes: Optional[bytes]):
    if mano_bytes is None:
        return {
            "error": "missing mano.npy",
        }
    try:
        array = np.load(io.BytesIO(mano_bytes), allow_pickle=False)
        decoded = decode_mano_sample_array(array)
    except Exception as error:
        return {
            "error": str(error),
        }

    flat = array.reshape(-1).astype(np.float32)
    preview_values = [round(float(value), 6) for value in flat[: min(24, flat.size)].tolist()]
    return {
        "shape": str(tuple(array.shape)),
        "dtype": str(array.dtype),
        "schema": MANO_SCHEMA,
        "preview": json.dumps(preview_values, ensure_ascii=False, indent=2),
        "array": array,
        "decoded": decoded,
    }


def rot6_to_rotmat(r6: np.ndarray) -> np.ndarray:
    a1 = r6[:3].astype(np.float32)
    a2 = r6[3:6].astype(np.float32)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def _decode_lowdim_fields(lowdim_array: np.ndarray) -> dict[str, np.ndarray]:
    flat = lowdim_array.reshape(-1).astype(np.float32)
    if flat.shape[0] < 116:
        raise ValueError(f"Lowdim vector too short for decode: {flat.shape[0]}")
    return {
        "left_wrist_world": flat[0:3],
        "right_wrist_world": flat[3:6],
        "left_root_rot6d": flat[6:12],
        "right_root_rot6d": flat[12:18],
        "left_fingertips_world": flat[18:33].reshape(5, 3),
        "right_fingertips_world": flat[33:48].reshape(5, 3),
        "camera_w2c": flat[96:112].reshape(4, 4),
        "camera_intrinsic": flat[112:116],
    }


def _presence_flags(presence: Optional[int]) -> tuple[bool, bool]:
    if presence is None:
        return True, True
    return presence in (1, 3), presence in (2, 3)


def _decode_image_bgr(image_bytes: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image bytes")
    return image


def _encode_image_jpeg(image_bgr: np.ndarray) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode rendered image")
    return encoded.tobytes()


def _extract_camera_from_lowdim(lowdim_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    decoded = _decode_lowdim_fields(lowdim_array)
    return decoded["camera_w2c"], decoded["camera_intrinsic"]


def _world_to_camera(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(c2w.astype(np.float32))
    return (w2c[:3, :3] @ points_world.T).T + w2c[:3, 3]


def _project_points(points_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray):
    pts_cam = _world_to_camera(points_world.astype(np.float32), c2w)
    fx, fy, cx, cy = [float(v) for v in intrinsic]
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((points_world.shape[0], 2), dtype=np.float32)
    uv[:, 0] = fx * (pts_cam[:, 0] / (z + 1e-8)) + cx
    uv[:, 1] = fy * (pts_cam[:, 1] / (z + 1e-8)) + cy
    return uv, valid


def _resolve_camera_c2w(extrinsic_raw: np.ndarray) -> tuple[np.ndarray, str]:
    raw = extrinsic_raw.astype(np.float32)
    try:
        return np.linalg.inv(raw), "w2c(fixed)"
    except np.linalg.LinAlgError as error:
        raise ValueError(f"Failed to invert WDS camera_w2c matrix: {error}") from error


def _apply_demo_rx_points(points_world: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32)
    if points.ndim == 1:
        return DEMO_RX_3X3 @ points
    return (DEMO_RX_3X3 @ points.T).T.astype(np.float32)


def _apply_demo_rx_rotmat(rotmat_world: np.ndarray) -> np.ndarray:
    rotmat = np.asarray(rotmat_world, dtype=np.float32).reshape(3, 3)
    return (DEMO_RX_3X3 @ rotmat).astype(np.float32)


def _apply_demo_rx_c2w(c2w: np.ndarray) -> np.ndarray:
    matrix = np.asarray(c2w, dtype=np.float32).reshape(4, 4).copy()
    matrix[:3, :3] = DEMO_RX_3X3 @ matrix[:3, :3]
    matrix[:3, 3] = DEMO_RX_3X3 @ matrix[:3, 3]
    return matrix


def _clip_uv_mask(uv: np.ndarray, valid: np.ndarray, image_shape) -> np.ndarray:
    h, w = image_shape[:2]
    return valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)


def _draw_axes(image_bgr: np.ndarray, origin_world: np.ndarray, rotmat_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray, axis_length: float = 0.04):
    import cv2

    origin_and_axes = np.stack(
        [
            origin_world,
            origin_world + rotmat_world[:, 0] * axis_length,
            origin_world + rotmat_world[:, 1] * axis_length,
            origin_world + rotmat_world[:, 2] * axis_length,
        ],
        axis=0,
    ).astype(np.float32)
    uv, valid = _project_points(origin_and_axes, c2w, intrinsic)
    mask = _clip_uv_mask(uv, valid, image_bgr.shape)
    if not mask[0]:
        return
    origin = tuple(uv[0].astype(np.int32))
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for axis_idx in range(3):
        if mask[axis_idx + 1]:
            cv2.arrowedLine(image_bgr, origin, tuple(uv[axis_idx + 1].astype(np.int32)), axis_colors[axis_idx], 2, tipLength=0.2)


def _render_keypoint_overlay(image_bytes: bytes, keypoint_frame: dict, presence: Optional[int]) -> np.ndarray:
    import cv2

    image_bgr = _decode_image_bgr(image_bytes)
    overlay = image_bgr.copy()
    c2w = keypoint_frame["c2w"]
    intrinsic = keypoint_frame["intrinsic"]
    left_present, right_present = _presence_flags(presence)

    for hand in keypoint_frame["hands"]:
        if hand["side"] == "left" and not left_present:
            continue
        if hand["side"] == "right" and not right_present:
            continue
        wrist = hand["wrist"]
        rotmat = hand["rotmat"]
        tips = hand["tips"]
        color = tuple(int(v) for v in hand["color"])
        tip_radius = int(hand.get("tip_radius", 4))
        wrist_radius = int(hand.get("wrist_radius", 6))
        line_thickness = int(hand.get("line_thickness", 2))
        draw_axes = bool(hand.get("draw_axes", False))
        label = hand.get("label")
        points_world = np.concatenate([wrist[None, :], tips], axis=0)
        uv, valid = _project_points(points_world, c2w, intrinsic)
        mask = _clip_uv_mask(uv, valid, image_bgr.shape)
        if mask[0]:
            wrist_uv = tuple(uv[0].astype(np.int32))
            cv2.circle(overlay, wrist_uv, wrist_radius, color, -1)
            if label:
                cv2.putText(
                    overlay,
                    label,
                    (wrist_uv[0] + 8, wrist_uv[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            for tip_idx in range(1, 6):
                if mask[tip_idx]:
                    tip_uv = tuple(uv[tip_idx].astype(np.int32))
                    cv2.line(overlay, wrist_uv, tip_uv, color, line_thickness)
                    cv2.circle(overlay, tip_uv, tip_radius, color, -1)
            if draw_axes:
                _draw_axes(overlay, wrist, rotmat, c2w, intrinsic)

    return cv2.addWeighted(overlay, 0.78, image_bgr, 0.22, 0)


def _build_mano_frame_from_sample(
    lowdim_array: np.ndarray,
    mano_array: np.ndarray,
    runtime: dict,
) -> dict:
    fields = _decode_lowdim_fields(lowdim_array)
    decoded = decode_mano_sample_array(mano_array)

    left_root = rot6d_to_axis_angle(fields["left_root_rot6d"])
    right_root = rot6d_to_axis_angle(fields["right_root_rot6d"])
    left_verts, left_joints = run_manopth_mano(
        runtime["mano_left"],
        wrist_world=fields["left_wrist_world"][None, :],
        root_rot_axis_angle=np.asarray(left_root, dtype=np.float32)[None, :],
        hand_pose_pca=decoded["left_pose_pca"][None, :],
        betas=decoded["left_betas"][None, :],
        device=runtime["device"],
    )
    right_verts, right_joints = run_manopth_mano(
        runtime["mano_right"],
        wrist_world=fields["right_wrist_world"][None, :],
        root_rot_axis_angle=np.asarray(right_root, dtype=np.float32)[None, :],
        hand_pose_pca=decoded["right_pose_pca"][None, :],
        betas=decoded["right_betas"][None, :],
        device=runtime["device"],
    )
    c2w, camera_convention = _resolve_camera_c2w(fields["camera_w2c"])
    return {
        "c2w": c2w,
        "intrinsic": fields["camera_intrinsic"],
        "camera_convention": camera_convention,
        "left_verts": np.asarray(left_verts[0], dtype=np.float32),
        "left_joints": np.asarray(left_joints[0], dtype=np.float32),
        "right_verts": np.asarray(right_verts[0], dtype=np.float32),
        "right_joints": np.asarray(right_joints[0], dtype=np.float32),
    }


def _draw_mano_hand(image_bgr: np.ndarray, overlay: np.ndarray, verts_world: np.ndarray, joints_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray, point_color, joint_color):
    import cv2

    verts_uv, verts_valid = _project_points(verts_world, c2w, intrinsic)
    verts_mask = _clip_uv_mask(verts_uv, verts_valid, image_bgr.shape)
    for pt in verts_uv[verts_mask].astype(np.int32):
        cv2.circle(overlay, tuple(pt), 1, point_color, -1)

    joints_uv, joints_valid = _project_points(joints_world, c2w, intrinsic)
    joints_mask = _clip_uv_mask(joints_uv, joints_valid, image_bgr.shape)
    for chain in MANO_JOINT_TREE:
        for j1, j2 in chain:
            if joints_mask[j1] and joints_mask[j2]:
                cv2.line(overlay, tuple(joints_uv[j1].astype(np.int32)), tuple(joints_uv[j2].astype(np.int32)), joint_color, 2)
    for idx, pt in enumerate(joints_uv.astype(np.int32)):
        if joints_mask[idx]:
            cv2.circle(overlay, tuple(pt), 3 if idx else 5, joint_color, -1)


def _render_mano_overlay(image_bytes: bytes, c2w: np.ndarray, intrinsic: np.ndarray, mano_frame: dict, presence: Optional[int]) -> np.ndarray:
    import cv2

    image_bgr = _decode_image_bgr(image_bytes)
    overlay = image_bgr.copy()
    left_present, right_present = _presence_flags(presence)

    if left_present and mano_frame.get("left_verts") is not None:
        _draw_mano_hand(
            image_bgr,
            overlay,
            mano_frame["left_verts"],
            mano_frame["left_joints"],
            c2w,
            intrinsic,
            point_color=(255, 255, 0),
            joint_color=(255, 180, 0),
        )
    if right_present and mano_frame.get("right_verts") is not None:
        _draw_mano_hand(
            image_bgr,
            overlay,
            mano_frame["right_verts"],
            mano_frame["right_joints"],
            c2w,
            intrinsic,
            point_color=(0, 255, 255),
            joint_color=(0, 200, 255),
        )

    return cv2.addWeighted(overlay, 0.72, image_bgr, 0.28, 0)


class ViewerApp:
    def __init__(self, args):
        tar_paths = resolve_tar_paths(args.input)
        summaries = scan_samples(
            tar_paths,
            sample_limit=args.sample_limit,
            episode_limit=None if args.all_episodes else args.episode_limit,
            filter_key=args.filter_key,
            filter_presence=args.filter_presence,
        )
        if args.shuffle:
            random.Random(42).shuffle(summaries)

        self.summaries = []
        self.summary_by_id = {}
        for new_id, summary in enumerate(summaries):
            updated = SampleSummary(**{**asdict(summary), "id": new_id})
            self.summaries.append(updated)
            self.summary_by_id[updated.id] = updated

        self.host = args.host
        self.port = args.port
        self.input_path = str(Path(args.input).expanduser().resolve())
        self.tar_paths = tar_paths
        self.start_index = min(max(args.start_index, 0), max(len(self.summaries) - 1, 0))
        self.episode_keys = sorted({summary.episode_key for summary in self.summaries})
        self.default_render_mode = args.render_mode
        self.apply_demo_rx = bool(args.apply_demo_rx)
        self.keypoint_source = args.keypoint_source
        self.processed_root = args.processed_root
        self.mano_dir = args.mano_dir
        self.mano_device = args.mano_device
        self._legacy_episode_cache = None
        self._demo_source_cache: dict[str, dict] = {}
        self._demo_mano_models = None
        self._mano_runtime = None
        self._mano_sample_cache: dict[int, dict] = {}
        self._cache_lock = threading.RLock()
        self._sample_keys_by_shard: dict[str, set[str]] = {}
        for summary in self.summaries:
            self._sample_keys_by_shard.setdefault(summary.shard_path, set()).add(summary.key)
        self._sample_cache: dict[tuple[str, str], dict] = {}
        self._loaded_shards: set[str] = set()
        self._render_bgr_cache: dict[tuple[int, str], np.ndarray] = {}
        self._render_jpeg_cache: dict[tuple[int, str], bytes] = {}
        self._sample_payload_cache: dict[tuple[int, str], dict] = {}
        self._cache_epoch = 0

    def clear_runtime_caches(self) -> None:
        with self._cache_lock:
            self._sample_cache.clear()
            self._loaded_shards.clear()
            self._render_bgr_cache.clear()
            self._render_jpeg_cache.clear()
            self._sample_payload_cache.clear()
            self._mano_sample_cache.clear()
            self._cache_epoch += 1

    def _load_legacy_episode_cache(self) -> list[dict]:
        if self._legacy_episode_cache is not None:
            return self._legacy_episode_cache
        if not self.processed_root:
            raise ValueError(
                "processed root not configured; rerun with --processed-root <legacy_processed_root>"
            )
        cache_path = Path(self.processed_root).expanduser().resolve() / "_vla_episodes_cache.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"Legacy episode cache not found: {cache_path}")
        self._legacy_episode_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        return self._legacy_episode_cache

    def _resolve_demo_seq_folder(self, meta: Optional[dict]) -> str:
        if not meta:
            raise ValueError("meta.json is missing; cannot resolve demo source episode")
        episode_index = meta.get("episode_index")
        if episode_index is None:
            raise ValueError(
                "Unable to resolve source episode: need meta.episode_index with --processed-root"
            )
        episodes = self._load_legacy_episode_cache()
        episode_idx = int(episode_index)
        if episode_idx < 0 or episode_idx >= len(episodes):
            raise IndexError(f"episode_index out of range for legacy cache: {episode_idx}")
        return str(episodes[episode_idx]["crop_dir"])

    def _ensure_demo_mano_models(self):
        if self._demo_mano_models is not None:
            return self._demo_mano_models

        import torch
        from lib.models.mano_wrapper import MANO

        use_cuda = str(self.mano_device).startswith("cuda") and torch.cuda.is_available()
        mano_root = str(Path(self.mano_dir or DEFAULT_MANO_DIR).expanduser().resolve())
        right_cfg = {
            "data_dir": mano_root,
            "model_path": mano_root,
            "gender": "neutral",
            "num_hand_joints": 15,
            "create_body_pose": False,
        }
        left_cfg = {
            "data_dir": mano_root,
            "model_path": mano_root,
            "gender": "neutral",
            "num_hand_joints": 15,
            "create_body_pose": False,
            "is_rhand": False,
        }
        mano_right = MANO(**right_cfg)
        mano_left = MANO(**left_cfg)
        mano_left.shapedirs[:, 0, :] *= -1
        if use_cuda:
            mano_right = mano_right.cuda()
            mano_left = mano_left.cuda()
        self._demo_mano_models = {
            "use_cuda": use_cuda,
            "mano_right": mano_right,
            "mano_left": mano_left,
        }
        return self._demo_mano_models

    def _load_demo_source_episode(self, seq_folder: str) -> dict:
        with self._cache_lock:
            cached = self._demo_source_cache.get(seq_folder)
        if cached is not None:
            return cached

        import joblib
        import numpy as np
        from hawor.utils.process import run_mano, run_mano_left
        from lib.eval_utils.custom_utils import load_slam_cam

        world_res_path = Path(seq_folder) / "world_space_res.pth"
        if not world_res_path.exists():
            raise FileNotFoundError(f"world_space_res.pth not found: {world_res_path}")
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res_path)
        del pred_valid

        models = self._ensure_demo_mano_models()
        outputs_left = run_mano_left(
            pred_trans[0:1],
            pred_rot[0:1],
            pred_hand_pose[0:1],
            None,
            pred_betas[0:1],
            use_cuda=models["use_cuda"],
            mano_model=models["mano_left"],
        )
        outputs_right = run_mano(
            pred_trans[1:2],
            pred_rot[1:2],
            pred_hand_pose[1:2],
            None,
            pred_betas[1:2],
            use_cuda=models["use_cuda"],
            mano_model=models["mano_right"],
        )
        left_joints = outputs_left["joints"][0].detach().cpu().numpy().astype(np.float32)
        right_joints = outputs_right["joints"][0].detach().cpu().numpy().astype(np.float32)

        slam_files = sorted((Path(seq_folder) / "SLAM").glob("hawor_slam_w_scale_*.npz"))
        if not slam_files:
            raise FileNotFoundError(f"No hawor_slam_w_scale_*.npz found under {(Path(seq_folder) / 'SLAM')}")
        r_w2c, t_w2c, r_c2w, t_c2w = load_slam_cam(str(slam_files[0]))
        del r_w2c, t_w2c
        r_x = DEMO_RX_3X3
        r_c2w = np.einsum("ij,njk->nik", r_x, r_c2w.cpu().numpy()).astype(np.float32)
        t_c2w = np.einsum("ij,nj->ni", r_x, t_c2w.cpu().numpy()).astype(np.float32)
        left_joints = np.einsum("ij,tnj->tni", r_x, left_joints).astype(np.float32)
        right_joints = np.einsum("ij,tnj->tni", r_x, right_joints).astype(np.float32)

        est_focal_path = Path(seq_folder) / "est_focal.txt"
        focal = None
        if est_focal_path.exists():
            try:
                focal = float(est_focal_path.read_text(encoding="utf-8").strip())
            except ValueError:
                focal = None
        if focal is None:
            slam_npz = np.load(str(slam_files[0]), allow_pickle=True)
            focal = float(slam_npz.get("img_focal", 600.0))

        demo_source = {
            "left_joints": left_joints,
            "right_joints": right_joints,
            "r_c2w": r_c2w,
            "t_c2w": t_c2w,
            "focal": float(focal),
        }
        with self._cache_lock:
            self._demo_source_cache[seq_folder] = demo_source
        return demo_source

    def _get_mano_frame(
        self,
        summary: SampleSummary,
        lowdim_array: np.ndarray,
        mano_array: np.ndarray,
        meta: Optional[dict] = None,
    ) -> dict:
        with self._cache_lock:
            cached = self._mano_sample_cache.get(summary.id)
        if cached is not None:
            return cached

        frame = _build_mano_frame_from_sample(
            lowdim_array,
            mano_array,
            self._ensure_mano_runtime(),
        )
        if self.apply_demo_rx:
            frame["c2w"] = _apply_demo_rx_c2w(frame["c2w"])
            frame["left_verts"] = _apply_demo_rx_points(frame["left_verts"])
            frame["left_joints"] = _apply_demo_rx_points(frame["left_joints"])
            frame["right_verts"] = _apply_demo_rx_points(frame["right_verts"])
            frame["right_joints"] = _apply_demo_rx_points(frame["right_joints"])
        frame["frame_idx"] = parse_frame_index(summary.key)
        with self._cache_lock:
            self._mano_sample_cache[summary.id] = frame
        return frame

    def _build_demo_keypoint_frame(self, summary: SampleSummary, sample: dict, meta: Optional[dict]) -> dict:
        seq_folder = self._resolve_demo_seq_folder(meta)
        source = self._load_demo_source_episode(seq_folder)
        frame_idx = parse_frame_index(summary.key)
        if frame_idx >= source["left_joints"].shape[0] or frame_idx >= source["r_c2w"].shape[0]:
            raise IndexError(
                f"frame_idx={frame_idx} out of range for source episode {seq_folder}"
            )
        image_bgr = _decode_image_bgr(sample["image_bytes"])
        height, width = image_bgr.shape[:2]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = source["r_c2w"][frame_idx]
        c2w[:3, 3] = source["t_c2w"][frame_idx]
        intrinsic = np.array(
            [source["focal"], source["focal"], width / 2.0, height / 2.0],
            dtype=np.float32,
        )
        return {
            "c2w": c2w,
            "intrinsic": intrinsic,
            "left_wrist": np.asarray(source["left_joints"][frame_idx, MANO_CENTER_IDX], dtype=np.float32),
            "right_wrist": np.asarray(source["right_joints"][frame_idx, MANO_CENTER_IDX], dtype=np.float32),
            "left_tips": np.asarray(source["left_joints"][frame_idx, FINGERTIP_INDICES], dtype=np.float32),
            "right_tips": np.asarray(source["right_joints"][frame_idx, FINGERTIP_INDICES], dtype=np.float32),
        }

    def _build_keypoint_frame(
        self,
        summary: SampleSummary,
        sample: dict,
        meta: Optional[dict],
        lowdim_array: np.ndarray,
        mano_array: Optional[np.ndarray] = None,
    ) -> dict:
        fields = _decode_lowdim_fields(lowdim_array)
        notes = [
            "All 3D lowdim fields are stored in the HaWoR/SLAM world frame."
        ]
        if self.apply_demo_rx:
            notes.append("Applied demo-style Rx=diag(1,-1,-1) to world points and camera pose for display.")
        keypoint_frame = {
            "c2w": fields["camera_w2c"],
            "intrinsic": fields["camera_intrinsic"],
            "left_wrist": fields["left_wrist_world"],
            "right_wrist": fields["right_wrist_world"],
            "left_tips": fields["left_fingertips_world"],
            "right_tips": fields["right_fingertips_world"],
            "left_rotmat": rot6_to_rotmat(fields["left_root_rot6d"]),
            "right_rotmat": rot6_to_rotmat(fields["right_root_rot6d"]),
            "anchor_source": "lowdim_wrist_world",
            "notes": notes,
        }

        points_world = np.concatenate(
            [
                keypoint_frame["left_wrist"][None, :],
                keypoint_frame["right_wrist"][None, :],
                keypoint_frame["left_tips"],
                keypoint_frame["right_tips"],
            ],
            axis=0,
        )
        keypoint_frame["c2w"], camera_convention = _resolve_camera_c2w(fields["camera_w2c"])
        notes.append("camera extrinsic is interpreted as fixed world-to-camera (w2c) and inverted for display.")
        if meta is not None and meta.get("camera_extrinsic_convention") is not None:
            notes.append(
                f"meta.json declares camera_extrinsic_convention={meta.get('camera_extrinsic_convention')!r}; visualizer now always uses w2c."
            )

        if self.apply_demo_rx:
            keypoint_frame["c2w"] = _apply_demo_rx_c2w(keypoint_frame["c2w"])
            keypoint_frame["left_wrist"] = _apply_demo_rx_points(keypoint_frame["left_wrist"])
            keypoint_frame["right_wrist"] = _apply_demo_rx_points(keypoint_frame["right_wrist"])
            keypoint_frame["left_tips"] = _apply_demo_rx_points(keypoint_frame["left_tips"])
            keypoint_frame["right_tips"] = _apply_demo_rx_points(keypoint_frame["right_tips"])
            keypoint_frame["left_rotmat"] = _apply_demo_rx_rotmat(keypoint_frame["left_rotmat"])
            keypoint_frame["right_rotmat"] = _apply_demo_rx_rotmat(keypoint_frame["right_rotmat"])

        lowdim_hands = [
            {
                "side": "left",
                "wrist": np.asarray(keypoint_frame["left_wrist"], dtype=np.float32),
                "tips": np.asarray(keypoint_frame["left_tips"], dtype=np.float32),
                "rotmat": np.asarray(keypoint_frame["left_rotmat"], dtype=np.float32),
                "color": (255, 0, 255),
                "label": "L-lowdim",
                "draw_axes": self.keypoint_source in ("auto", "lowdim"),
            },
            {
                "side": "right",
                "wrist": np.asarray(keypoint_frame["right_wrist"], dtype=np.float32),
                "tips": np.asarray(keypoint_frame["right_tips"], dtype=np.float32),
                "rotmat": np.asarray(keypoint_frame["right_rotmat"], dtype=np.float32),
                "color": (0, 255, 0),
                "label": "R-lowdim",
                "draw_axes": self.keypoint_source in ("auto", "lowdim"),
            },
        ]
        keypoint_frame["hands"] = lowdim_hands

        if self.keypoint_source in ("demo", "compare-demo"):
            demo_frame = self._build_demo_keypoint_frame(summary, sample, meta)
            if not self.apply_demo_rx:
                for hand in lowdim_hands:
                    hand["wrist"] = _apply_demo_rx_points(hand["wrist"])
                    hand["tips"] = _apply_demo_rx_points(hand["tips"])
                    hand["rotmat"] = _apply_demo_rx_rotmat(hand["rotmat"])
            demo_hands = [
                {
                    "side": "left",
                    "wrist": np.asarray(demo_frame["left_wrist"], dtype=np.float32),
                    "tips": np.asarray(demo_frame["left_tips"], dtype=np.float32),
                    "rotmat": np.asarray(keypoint_frame["left_rotmat"], dtype=np.float32),
                    "color": (255, 255, 0),
                    "label": "L-demo",
                    "draw_axes": self.keypoint_source == "demo",
                },
                {
                    "side": "right",
                    "wrist": np.asarray(demo_frame["right_wrist"], dtype=np.float32),
                    "tips": np.asarray(demo_frame["right_tips"], dtype=np.float32),
                    "rotmat": np.asarray(keypoint_frame["right_rotmat"], dtype=np.float32),
                    "color": (0, 255, 255),
                    "label": "R-demo",
                    "draw_axes": self.keypoint_source == "demo",
                },
            ]
            keypoint_frame["c2w"] = np.asarray(demo_frame["c2w"], dtype=np.float32)
            keypoint_frame["intrinsic"] = np.asarray(demo_frame["intrinsic"], dtype=np.float32)
            if self.keypoint_source == "demo":
                keypoint_frame["left_wrist"] = demo_hands[0]["wrist"]
                keypoint_frame["right_wrist"] = demo_hands[1]["wrist"]
                keypoint_frame["left_tips"] = demo_hands[0]["tips"]
                keypoint_frame["right_tips"] = demo_hands[1]["tips"]
                keypoint_frame["hands"] = demo_hands
                keypoint_frame["anchor_source"] = "demo_offline_source"
                notes.append(
                    "keypoint_source=demo: draw direct source joints using the demo_offline camera and image-center projection."
                )
                return keypoint_frame

            for hand in lowdim_hands:
                hand["wrist_radius"] = 7
                hand["tip_radius"] = 5
                hand["line_thickness"] = 3
                hand["draw_axes"] = False
            for hand in demo_hands:
                hand["wrist_radius"] = 4
                hand["tip_radius"] = 3
                hand["line_thickness"] = 2
                hand["draw_axes"] = False
            keypoint_frame["hands"] = lowdim_hands + demo_hands
            keypoint_frame["anchor_source"] = "compare_lowdim_vs_demo"
            notes.append(
                "keypoint_source=compare-demo: lowdim is drawn in magenta/green, demo_offline source is drawn in yellow/cyan."
            )
            notes.append(
                "Both overlays use the demo_offline-aligned camera and image-center projection to isolate keypoint generation differences."
            )
            return keypoint_frame

        mano_frame = None
        mano_error = None
        if mano_array is None:
            mano_error = "mano.npy is missing"
        else:
            try:
                mano_frame = self._get_mano_frame(summary, lowdim_array, mano_array, meta=meta)
            except Exception as error:
                mano_error = str(error)

        if mano_frame is None and self.keypoint_source in ("mano", "compare"):
            notes.append(
                f"Requested keypoint_source={self.keypoint_source}, but MANO decode is unavailable ({mano_error}); falling back to lowdim."
            )
            return keypoint_frame
        elif mano_frame is None:
            notes.append(
                "MANO is unavailable for this frame, so keypoint mode uses lowdim wrist/tip world coordinates directly."
            )
            return keypoint_frame

        mano_hands = [
            {
                "side": "left",
                "wrist": np.asarray(mano_frame["left_joints"][MANO_CENTER_IDX], dtype=np.float32),
                "tips": np.asarray(mano_frame["left_joints"][FINGERTIP_INDICES], dtype=np.float32),
                "rotmat": np.asarray(keypoint_frame["left_rotmat"], dtype=np.float32),
                "color": (255, 255, 0),
                "label": "L-mano",
                "draw_axes": self.keypoint_source == "mano",
            },
            {
                "side": "right",
                "wrist": np.asarray(mano_frame["right_joints"][MANO_CENTER_IDX], dtype=np.float32),
                "tips": np.asarray(mano_frame["right_joints"][FINGERTIP_INDICES], dtype=np.float32),
                "rotmat": np.asarray(keypoint_frame["right_rotmat"], dtype=np.float32),
                "color": (0, 255, 255),
                "label": "R-mano",
                "draw_axes": self.keypoint_source == "mano",
            },
        ]

        if self.keypoint_source == "lowdim":
            notes.append("keypoint_source=lowdim: draw raw lowdim wrist/tip positions only.")
            return keypoint_frame

        if self.keypoint_source == "compare":
            for hand in lowdim_hands:
                hand["wrist_radius"] = 7
                hand["tip_radius"] = 5
                hand["line_thickness"] = 3
                hand["draw_axes"] = False
            for hand in mano_hands:
                hand["wrist_radius"] = 4
                hand["tip_radius"] = 3
                hand["line_thickness"] = 2
                hand["draw_axes"] = False
            keypoint_frame["hands"] = lowdim_hands + mano_hands
            keypoint_frame["anchor_source"] = "compare_lowdim_vs_mano"
            notes.append(
                "keypoint_source=compare: lowdim is drawn in magenta/green, MANO is drawn in yellow/cyan."
            )
            return keypoint_frame

        if self.keypoint_source in ("mano", "auto"):
            keypoint_frame["left_wrist"] = mano_hands[0]["wrist"]
            keypoint_frame["right_wrist"] = mano_hands[1]["wrist"]
            keypoint_frame["left_tips"] = mano_hands[0]["tips"]
            keypoint_frame["right_tips"] = mano_hands[1]["tips"]
            keypoint_frame["hands"] = mano_hands
            keypoint_frame["anchor_source"] = "mano_joint_wrist"
            notes.append(
                "Keypoint mode is using MANO joint 0 as the wrist anchor and MANO fingertip joints."
            )
        return keypoint_frame

    def _ensure_mano_runtime(self):
        if self._mano_runtime is not None:
            return self._mano_runtime

        import torch

        requested = self.mano_device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print(f"MANO device {requested} requested but CUDA is unavailable; falling back to cpu.", flush=True)
            requested = "cpu"
        device = torch.device(requested)
        print(f"Initializing MANO runtime on {device} ...", flush=True)
        mano_right, mano_left = build_manopth_models(
            device,
            mano_dir=self.mano_dir,
            center_idx=MANO_CENTER_IDX,
            flat_hand_mean=MANO_FLAT_HAND_MEAN,
            ncomps=MANO_PCA_DIMS,
        )
        self._mano_runtime = {
            "device": device,
            "mano_right": mano_right,
            "mano_left": mano_left,
        }
        return self._mano_runtime

    def _ensure_shard_samples_loaded(self, shard_path: str):
        with self._cache_lock:
            if shard_path in self._loaded_shards:
                return
            target_keys = self._sample_keys_by_shard.get(shard_path, set())
        if not target_keys:
            with self._cache_lock:
                self._loaded_shards.add(shard_path)
            return

        loaded = {}
        for sample in iter_shard_samples(shard_path):
            sample_key = sample["key"]
            if sample_key in target_keys:
                loaded[(shard_path, sample_key)] = sample
                if len(loaded) >= len(target_keys):
                    break

        missing_keys = sorted(target_keys - {sample_key for _, sample_key in loaded})
        if missing_keys:
            preview = ", ".join(missing_keys[:3])
            raise KeyError(f"Failed to load {len(missing_keys)} indexed sample(s) from {shard_path}: {preview}")

        with self._cache_lock:
            self._sample_cache.update(loaded)
            self._loaded_shards.add(shard_path)

    def _get_sample(self, summary: SampleSummary) -> dict:
        cache_key = (summary.shard_path, summary.key)
        with self._cache_lock:
            cached = self._sample_cache.get(cache_key)
        if cached is not None:
            return cached
        self._ensure_shard_samples_loaded(summary.shard_path)
        with self._cache_lock:
            cached = self._sample_cache.get(cache_key)
        if cached is None:
            raise KeyError(f"Sample key not found in shard {summary.shard_path}: {summary.key}")
        return cached

    def _render_image(self, *, render_mode: str, summary: SampleSummary, sample: dict, lowdim_array: Optional[np.ndarray], mano_array: Optional[np.ndarray], presence: Optional[int]) -> np.ndarray:
        if sample["image_bytes"] is None:
            raise ValueError("missing image.jpg")

        if render_mode == "keypoint":
            if lowdim_array is None:
                raise ValueError("lowdim.npy is required for keypoint render")
            meta, _ = decode_meta(sample["meta_bytes"])
            keypoint_frame = self._build_keypoint_frame(
                summary,
                sample,
                meta,
                lowdim_array,
                mano_array=mano_array,
            )
            return _render_keypoint_overlay(sample["image_bytes"], keypoint_frame, presence)

        if render_mode == "mano":
            if lowdim_array is None:
                raise ValueError("lowdim.npy is required for mano render")
            if mano_array is None:
                raise ValueError("mano.npy is required for mano render")
            meta, _ = decode_meta(sample["meta_bytes"])
            mano_frame = self._get_mano_frame(summary, lowdim_array, mano_array, meta=meta)
            return _render_mano_overlay(sample["image_bytes"], mano_frame["c2w"], mano_frame["intrinsic"], mano_frame, presence)

        raise ValueError(f"Unsupported render mode: {render_mode}")

    def rendered_image_bgr(self, sample_id: int, render_mode: str) -> np.ndarray:
        if sample_id not in self.summary_by_id:
            raise KeyError(f"Unknown sample id: {sample_id}")

        render_cache_key = (sample_id, render_mode)
        with self._cache_lock:
            cached = self._render_bgr_cache.get(render_cache_key)
        if cached is not None:
            return cached

        summary = self.summary_by_id[sample_id]
        sample = self._get_sample(summary)
        meta, _ = decode_meta(sample["meta_bytes"])
        lowdim_summary = summarize_lowdim(sample["lowdim_bytes"])
        lowdim_array = lowdim_summary.get("array")
        mano_summary = summarize_mano(sample.get("mano_bytes"))
        mano_array = mano_summary.get("array")
        image_bgr = self._render_image(
            render_mode=render_mode,
            summary=summary,
            sample=sample,
            lowdim_array=lowdim_array,
            mano_array=mano_array,
            presence=None if meta is None else meta.get("presence"),
        )
        with self._cache_lock:
            self._render_bgr_cache[render_cache_key] = image_bgr
        return image_bgr

    def rendered_image_bytes(self, sample_id: int, render_mode: str) -> bytes:
        if sample_id not in self.summary_by_id:
            raise KeyError(f"Unknown sample id: {sample_id}")

        render_cache_key = (sample_id, render_mode)
        with self._cache_lock:
            cached = self._render_jpeg_cache.get(render_cache_key)
        if cached is not None:
            return cached

        image_bgr = self.rendered_image_bgr(sample_id, render_mode)
        image_bytes = _encode_image_jpeg(image_bgr)
        with self._cache_lock:
            self._render_jpeg_cache[render_cache_key] = image_bytes
        return image_bytes

    def summaries_grouped_by_episode(self) -> list[tuple[str, list[SampleSummary]]]:
        grouped: dict[str, list[SampleSummary]] = {}
        for summary in self.summaries:
            grouped.setdefault(summary.episode_key, []).append(summary)

        ordered = []
        for episode_key in self.episode_keys:
            items = grouped.get(episode_key, [])
            items.sort(key=lambda summary: parse_frame_index(summary.key))
            if items:
                ordered.append((episode_key, items))
        return ordered

    def export_videos(self, *, render_mode: str, output_path: Optional[str], fps: int) -> list[str]:
        import cv2
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        groups = self.summaries_grouped_by_episode()
        if not groups:
            raise ValueError("No episodes available for video export")

        output_root = Path(output_path).expanduser().resolve() if output_path else (Path.cwd() / "webdataset_visualizer_videos")
        outputs: list[str] = []

        if len(groups) == 1 and output_root.suffix.lower() == ".mp4":
            output_root.parent.mkdir(parents=True, exist_ok=True)
            targets = [(groups[0][0], groups[0][1], output_root)]
        else:
            if output_root.suffix:
                raise ValueError(
                    "When exporting multiple episodes, --video-out must be a directory or be omitted"
                )
            output_root.mkdir(parents=True, exist_ok=True)
            suffix = "demo_rx" if self.apply_demo_rx else "raw"
            targets = [
                (
                    episode_key,
                    summaries,
                    output_root / f"{sanitize_filename(episode_key)}.{render_mode}.{suffix}.mp4",
                )
                for episode_key, summaries in groups
            ]

        for episode_key, summaries, target_path in targets:
            first_frame = self.rendered_image_bgr(summaries[0].id, render_mode)
            height, width = first_frame.shape[:2]
            writer = cv2.VideoWriter(
                str(target_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {target_path}")
            try:
                writer.write(first_frame)
                iterator = summaries[1:]
                if tqdm is not None:
                    iterator = tqdm(
                        iterator,
                        desc=f"Render video {episode_key}",
                        unit="frame",
                    )
                for summary in iterator:
                    frame = self.rendered_image_bgr(summary.id, render_mode)
                    if frame.shape[:2] != (height, width):
                        raise ValueError(
                            f"Inconsistent frame size in episode {episode_key}: "
                            f"expected {(height, width)}, got {frame.shape[:2]}"
                        )
                    writer.write(frame)
            finally:
                writer.release()
            outputs.append(str(target_path))

        return outputs

    def config_payload(self):
        return {
            "input_path": self.input_path,
            "tar_count": len(self.tar_paths),
            "sample_count": len(self.summaries),
            "episode_count": len(self.episode_keys),
            "episode_keys": self.episode_keys,
            "host": self.host,
            "port": self.port,
            "start_index": self.start_index,
            "default_render_mode": self.default_render_mode,
            "available_render_modes": ["keypoint", "mano"],
            "apply_demo_rx": self.apply_demo_rx,
            "keypoint_source": self.keypoint_source,
            "cache_token": str(self._cache_epoch),
        }

    def index_payload(self):
        return [asdict(summary) for summary in self.summaries]

    def sample_payload(self, sample_id: int, render_mode: str, warm_image: bool = True):
        if sample_id not in self.summary_by_id:
            raise KeyError(f"Unknown sample id: {sample_id}")

        payload_cache_key = (sample_id, f"{render_mode}:{'warm' if warm_image else 'cold'}")
        with self._cache_lock:
            cached_payload = self._sample_payload_cache.get(payload_cache_key)
        if cached_payload is not None:
            return cached_payload

        summary = self.summary_by_id[sample_id]
        sample = self._get_sample(summary)
        meta, meta_error = decode_meta(sample["meta_bytes"])
        instruction_display = normalize_instruction(meta)

        errors = []
        if summary.missing_fields:
            errors.append("missing payloads: " + ", ".join(summary.missing_fields))
        if meta_error is not None:
            errors.append(f"meta decode failed: {meta_error}")

        lowdim_summary = summarize_lowdim(sample["lowdim_bytes"])
        if "error" in lowdim_summary:
            errors.append(f"lowdim load failed: {lowdim_summary['error']}")
        mano_summary = summarize_mano(sample.get("mano_bytes"))
        if render_mode == "mano" and "error" in mano_summary:
            errors.append(f"mano load failed: {mano_summary['error']}")

        lowdim_array = lowdim_summary.get("array")
        mano_array = mano_summary.get("array")
        render_notes = []
        if lowdim_array is not None:
            try:
                if render_mode == "keypoint":
                    render_notes = self._build_keypoint_frame(
                        summary,
                        sample,
                        meta,
                        lowdim_array,
                        mano_array=mano_array,
                    )["notes"]
                elif render_mode == "mano":
                    render_notes = [
                        "MANO mode reconstructs vertices and joints from the per-sample mano.npy payload using manopth."
                    ]
                    if mano_array is None:
                        render_notes.append(
                            "This sample does not contain mano.npy, so MANO render cannot proceed."
                        )
            except Exception as error:
                errors.append(f"{render_mode} note generation failed: {error}")

        image_url = None
        try:
            image_url = f"/api/image/{summary.id}?render_mode={render_mode}"
            if warm_image:
                self.rendered_image_bytes(summary.id, render_mode)
        except Exception as error:
            errors.append(f"{render_mode} render failed: {error}")
            if sample["image_bytes"] is not None:
                image_url = f"/api/raw-image/{summary.id}"

        payload = {
            "id": summary.id,
            "key": summary.key,
            "shard_path": summary.shard_path,
            "shard_name": summary.shard_name,
            "clip_id": summary.clip_id,
            "render_mode": render_mode,
            "instruction_num": None if meta is None else meta.get("instruction_num"),
            "instruction_display": instruction_display,
            "presence": None if meta is None else meta.get("presence"),
            "broken": bool(errors) or summary.broken,
            "errors": errors,
            "image_url": image_url,
            "meta_pretty": json.dumps(meta, ensure_ascii=False, indent=2) if meta is not None else None,
            "lowdim": {
                "error": lowdim_summary.get("error"),
                "shape": lowdim_summary.get("shape"),
                "dtype": lowdim_summary.get("dtype"),
                "coordinate_system": lowdim_summary.get("coordinate_system"),
                "min": lowdim_summary.get("min"),
                "max": lowdim_summary.get("max"),
                "mean": lowdim_summary.get("mean"),
            },
            "mano": {
                "error": mano_summary.get("error"),
                "shape": mano_summary.get("shape"),
                "dtype": mano_summary.get("dtype"),
                "schema": mano_summary.get("schema"),
                "preview": mano_summary.get("preview"),
            },
            "render_notes": render_notes,
            "lowdim_preview": lowdim_summary.get("preview"),
            "lowdim_segments": lowdim_summary.get("segments", []),
        }
        with self._cache_lock:
            self._sample_payload_cache[payload_cache_key] = payload
        return payload


def build_parser():
    parser = argparse.ArgumentParser(description="Local web viewer for WebDataset shards")
    parser.add_argument("--input", required=True, help="Path to a .tar shard or directory containing .tar shards")
    parser.add_argument(
        "--output-mode",
        default="web",
        choices=["web", "video"],
        help="Use the interactive web viewer or export offline mp4 video(s) with the same renderer.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sample order after indexing")
    parser.add_argument("--sample-limit", type=int, default=None, help="Only index the first N matched samples")
    parser.add_argument("--episode-limit", type=int, default=1, help="Only index the first N matched episodes")
    parser.add_argument("--all-episodes", action="store_true", help="Index all matched episodes instead of stopping early")
    parser.add_argument("--filter-key", default="", help="Initial substring filter on key / clip_id / instruction")
    parser.add_argument("--filter-presence", type=int, default=None, choices=[0, 1, 2, 3], help="Initial presence filter")
    parser.add_argument("--start-index", type=int, default=0, help="Initial sample index to open")
    parser.add_argument(
        "--render-mode",
        default="keypoint",
        choices=["keypoint", "mano"],
        help="Initial render mode. keypoint is lightweight; mano is the geometry-accurate view.",
    )
    parser.add_argument(
        "--keypoint-source",
        default="auto",
        choices=["auto", "lowdim", "mano", "compare", "demo", "compare-demo"],
        help="For keypoint mode: use lowdim, MANO, demo_offline-aligned source points, or draw comparisons.",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default=None,
        help="Legacy processed root used to resolve source seq_folder for demo-aligned keypoint diagnostics.",
    )
    parser.add_argument("--mano-dir", type=str, default=None, help="Optional MANO model directory override")
    parser.add_argument("--mano-device", type=str, default="cpu", help="Device for MANO rendering, e.g. cpu or cuda:0")
    parser.add_argument(
        "--video-out",
        type=str,
        default=None,
        help="Offline video output path. Single episode: may be a .mp4 file; multiple episodes: use a directory.",
    )
    parser.add_argument("--video-fps", type=int, default=30, help="FPS for offline video export")
    parser.add_argument(
        "--apply-demo-rx",
        action="store_true",
        help="Apply the same Rx=diag(1,-1,-1) world/camera transform used by demo_offline before rendering.",
    )
    return parser


def discover_access_urls(bind_host: str, port: int) -> list[str]:
    urls = []
    seen = set()

    def add_url(host: str):
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


def make_handler(app: ViewerApp):
    class Handler(BaseHTTPRequestHandler):
        def _send_no_cache_headers(self):
            # Force browsers/proxies to always fetch fresh content on reload.
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

        def _send_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_no_cache_headers()
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_bytes(self, body: bytes, content_type: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._send_no_cache_headers()
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
            self._send_no_cache_headers()
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
            if parsed.path == "/api/config":
                app.clear_runtime_caches()
                self._send_json(app.config_payload())
                return
            if parsed.path == "/api/index":
                self._send_json(app.index_payload())
                return
            if parsed.path.startswith("/api/sample/"):
                sample_id_str = parsed.path.rsplit("/", 1)[-1]
                try:
                    sample_id = int(sample_id_str)
                    render_mode = query.get("render_mode", [app.default_render_mode])[0]
                    warm_image = query.get("warm_image", ["1"])[0] != "0"
                    payload = app.sample_payload(
                        sample_id,
                        render_mode=render_mode,
                        warm_image=warm_image,
                    )
                except Exception as error:
                    self._send_json({"error": str(error)}, status=404)
                    return
                self._send_json(payload)
                return
            if parsed.path.startswith("/api/image/"):
                sample_id_str = parsed.path.rsplit("/", 1)[-1]
                try:
                    sample_id = int(sample_id_str)
                    render_mode = query.get("render_mode", [app.default_render_mode])[0]
                    body = app.rendered_image_bytes(sample_id, render_mode=render_mode)
                except Exception as error:
                    self._send_json({"error": str(error)}, status=404)
                    return
                self._send_bytes(body, "image/jpeg")
                return
            if parsed.path.startswith("/api/raw-image/"):
                sample_id_str = parsed.path.rsplit("/", 1)[-1]
                try:
                    sample_id = int(sample_id_str)
                    summary = app.summary_by_id[sample_id]
                    sample = app._get_sample(summary)
                    body = sample["image_bytes"]
                    if body is None:
                        raise ValueError("missing image.jpg")
                except Exception as error:
                    self._send_json({"error": str(error)}, status=404)
                    return
                self._send_bytes(body, "image/jpeg")
                return
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self._send_json({"error": f"Unsupported path: {parsed.path}"}, status=404)

    return Handler


def main():
    args = build_parser().parse_args()
    app = ViewerApp(args)
    if not app.summaries:
        raise SystemExit("No samples matched the current filters.")

    if args.output_mode == "video":
        outputs = app.export_videos(
            render_mode=args.render_mode,
            output_path=args.video_out,
            fps=args.video_fps,
        )
        print(
            f"Exported {len(outputs)} video(s) from {len(app.episode_keys)} episode(s) "
            f"using render_mode={args.render_mode} apply_demo_rx={app.apply_demo_rx}"
        )
        for path in outputs:
            print(f"Video output: {path}")
        return

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(
        f"Indexed {len(app.summaries)} sample(s) from {len(app.episode_keys)} episode(s) "
        f"across {len(app.tar_paths)} tar(s)"
    )
    print(f"Server bind: {args.host}:{args.port}")
    for url in discover_access_urls(args.host, args.port):
        print(f"Viewer URL: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
