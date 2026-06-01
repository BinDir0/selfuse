#!/usr/bin/env bash
# End-to-end smoke test for the HaWoR pipeline.
#
# WHY: the unit tests run on CPU, in isolation, with heavy mocking. They verify
# per-function logic but STRUCTURALLY cannot catch integration bugs that only
# appear with a real worker fork + real CUDA + the real manifest flow, e.g.:
#   - "Cannot re-initialize CUDA in forked subprocess" (parent touched CUDA pre-fork)
#   - descriptor-manifest input wiring (clip-ids vs real paths)
#   - preflight fail-fast actually aborting before GPU work
# This script runs the REAL pipeline and asserts invariants, so it catches that
# class of bug. Run it on a GPU machine after changes.
#
# USAGE (run the orchestrated + negative legs from the `hawor` conda env, which
# auto-resolves the any4d env for SLAM):
#   conda activate hawor
#   bash tools/ops/e2e_smoke.sh --video /abs/short.mp4 --tmp /efs-exp/$USER/tmp --gpu 0
#
# The optional standalone-batch leg (cleanup + resume checks) drives SLAM/Any4D
# directly, which needs the any4d env; enable it by passing that interpreter:
#   ... --batch-python "conda run --no-capture-output -n any4d python"
#
# Prefer a SHORT clip (faster). Exit code 0 = all assertions passed.
set -uo pipefail

VIDEO=""; TMP=""; GPU=0; OUT=""; PY="${HAWOR_E2E_PYTHON:-python}"; BATCH_PY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --tmp) TMP="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    --batch-python) BATCH_PY="$2"; shift 2;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1"; exit 64;;
  esac
done
[ -n "$VIDEO" ] && [ -n "$TMP" ] || { echo "usage: e2e_smoke.sh --video PATH --tmp DIR [--gpu N] [--out DIR] [--python P] [--batch-python P]"; exit 64; }
[ -f "$VIDEO" ] || { echo "video not found: $VIDEO"; exit 64; }
OUT="${OUT:-$(mktemp -d)}"; mkdir -p "$OUT" "$TMP"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export HAWOR_BATCH_TMPDIR="$TMP"

fails=0
pass(){ printf '  [PASS] %s\n' "$1"; }
fail(){ printf '  [FAIL] %s\n' "$1"; fails=$((fails + 1)); }

check_result_npz(){  # $1=seq_dir  $2=label  -> exit 0 if healthy
  "$PY" - "$1" "$2" <<'PY'
import sys, os, numpy as np
seq, label = sys.argv[1], sys.argv[2]
r = os.path.join(seq, "result.npz")
if not os.path.exists(r):
    print(f"    result.npz missing at {seq}"); sys.exit(1)
d = np.load(r, allow_pickle=True)
ok = True
need = {"pred_trans", "pred_rot", "pred_hand_pose", "pred_betas", "pred_valid"}
miss = need - set(d.files)
if miss:
    print(f"    [{label}] missing pose keys: {miss}"); ok = False
if "depths_uint16" not in d.files:
    print(f"    [{label}] no depth in result.npz"); ok = False
for k in need & set(d.files):
    a = np.asarray(d[k])
    if a.dtype.kind in "fc" and not np.isfinite(a).all():
        print(f"    [{label}] {k} has non-finite values"); ok = False
if "pred_valid" in d.files:
    vr = float(np.asarray(d["pred_valid"]).mean())
    print(f"    [{label}] valid-frame ratio = {vr:.3f}")
    if vr < 0.9:
        ok = False
sys.exit(0 if ok else 1)
PY
}

echo "== HaWoR e2e smoke =="
echo "repo=$REPO  video=$VIDEO  tmp=$TMP  gpu=$GPU  out=$OUT"

# ---------------------------------------------------------------------------
echo "[1] orchestrated prepare,infer (real fork + real CUDA + real manifest)"
ORCH="$OUT/orch"
cat > "$OUT/smoke_orch.yaml" <<EOF
video: $VIDEO
output_root: $ORCH
EOF
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_dataset_pipeline.py \
  --config "$OUT/smoke_orch.yaml" --stages prepare,infer --no-resume
rc=$?
[ "$rc" -eq 0 ] && pass "orchestrated exit 0" || fail "orchestrated exit=$rc"
ev="$(find "$ORCH" -name events.jsonl 2>/dev/null | head -1)"
if [ -n "$ev" ]; then
  nsf=$(grep -c 'stage_failure' "$ev" 2>/dev/null || true)
  [ "${nsf:-0}" -eq 0 ] && pass "no stage_failure events" || fail "$nsf stage_failure event(s): $ev"
fi
seqo="$(find "$ORCH/stage_outputs" -name result.npz -printf '%h\n' 2>/dev/null | head -1)"
if [ -n "$seqo" ]; then
  check_result_npz "$seqo" orchestrated && pass "orchestrated result.npz healthy (poses+depth)" || fail "orchestrated result.npz checks"
else
  fail "orchestrated result.npz not produced"
fi

# ---------------------------------------------------------------------------
echo "[2] preflight fail-fast (negative: must abort BEFORE GPU work, exit 2)"
echo "$VIDEO" > "$OUT/v.txt"
( unset HAWOR_BATCH_TMPDIR
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/batch_infer.py --video_list "$OUT/v.txt" \
    --gpus "$GPU" --stages slam --run_dir "$OUT/neg_tmp" >/dev/null 2>&1 )
rc=$?
[ "$rc" -eq 2 ] && pass "missing tmp root -> exit 2" || fail "missing tmp root: expected exit 2, got $rc"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/batch_infer.py --video_list "$OUT/v.txt" \
  --gpus "$GPU" --stages motion --checkpoint "$OUT/nope.ckpt" --run_dir "$OUT/neg_ckpt" >/dev/null 2>&1
rc=$?
[ "$rc" -eq 2 ] && pass "bad checkpoint -> exit 2" || fail "bad checkpoint: expected exit 2, got $rc"

# ---------------------------------------------------------------------------
if [ -n "$BATCH_PY" ]; then
  echo "[3] standalone batch: cleanup + resume (default keep_intermediates=none)"
  BOUT="$OUT/batch"
  CUDA_VISIBLE_DEVICES="$GPU" $BATCH_PY scripts/batch_infer.py --video_list "$OUT/v.txt" \
    --gpus "$GPU" --stages detect_track,motion,slam,infiller --slam_backend dpvo --any4d \
    --run_dir "$OUT/batch_run" --output_root "$BOUT"
  rc=$?
  [ "$rc" -eq 0 ] && pass "batch exit 0" || fail "batch exit=$rc"
  seqb="$(find "$BOUT" -name result.npz -printf '%h\n' 2>/dev/null | head -1)"
  if [ -n "$seqb" ]; then
    check_result_npz "$seqb" batch && pass "batch result.npz healthy" || fail "batch result.npz checks"
    for junk in cam_space extracted_images cam_space_cache.joblib; do
      [ -e "$seqb/$junk" ] && fail "cleanup: $junk still present" || pass "cleanup: $junk removed"
    done
    ls "$seqb"/tracks_* >/dev/null 2>&1 && fail "cleanup: tracks_* still present" || pass "cleanup: tracks_* removed"
    ls "$seqb"/.*.done  >/dev/null 2>&1 && fail "cleanup: .done markers present" || pass "cleanup: markers removed"
    m1=$(stat -c %Y "$seqb/result.npz")
    CUDA_VISIBLE_DEVICES="$GPU" $BATCH_PY scripts/batch_infer.py --video_list "$OUT/v.txt" \
      --gpus "$GPU" --stages detect_track,motion,slam,infiller --slam_backend dpvo --any4d \
      --run_dir "$OUT/batch_run2" --output_root "$BOUT" >/dev/null 2>&1
    m2=$(stat -c %Y "$seqb/result.npz")
    [ "$m1" = "$m2" ] && pass "resume skipped (result.npz unchanged)" || fail "resume re-ran (result.npz rewritten)"
  else
    fail "batch result.npz not produced"
  fi
else
  echo "[3] standalone batch cleanup/resume: SKIPPED (pass --batch-python to enable)"
fi

echo "== summary: $fails failure(s); artifacts in $OUT =="
[ "$fails" -eq 0 ] && { echo "E2E SMOKE: PASS"; exit 0; } || { echo "E2E SMOKE: FAIL ($fails)"; exit 1; }
