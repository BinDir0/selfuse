# Bug Fix: Stage Output Validation on Resume

## Problem

**Original Issue**: Test 4 in `test_smoke.sh` failed with "stage4 not regenerated" error.

**Root Cause**: When using `--resume`, the batch scheduler (`batch_infer.py`) only checked the in-memory status from `status.json` to decide whether to skip a stage. It did NOT verify that the actual output files still existed on disk.

**Failure Scenario**:
1. Run batch inference successfully (status.json records infiller="completed")
2. Manually delete `world_space_res.pth` (or any stage output)
3. Re-run with `--resume`
4. **Bug**: Scheduler sees status="completed" and skips the stage, even though output is missing
5. Result: Missing output is never regenerated

## Solution

Added `verify_stage_complete()` method to `batch_infer.py` that:
1. Checks status.json (in-memory state)
2. **AND** validates actual disk artifacts exist using `batch_worker.is_stage_complete()`
3. If status says "completed" but output is missing:
   - Emits `stage_revalidate` event with reason="output_missing"
   - Resets stage status to "pending"
   - Re-runs the stage

## Code Changes

**File**: `scripts/batch_infer.py`

**Added method** (line 147):
```python
def verify_stage_complete(self, video_path: str, stage: str) -> bool:
    """Verify that stage output actually exists on disk."""
    try:
        video_path_obj = Path(video_path)
        seq_folder = video_path_obj.parent / video_path_obj.stem

        # Import validation function from batch_worker
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from batch_worker import is_stage_complete

        return is_stage_complete(stage, seq_folder)
    except Exception:
        return False
```

**Modified logic** (line 166-181):
```python
for stage in self.stages:
    # Check both status.json AND actual disk artifacts
    if task.stage_status[stage] == "completed":
        if self.verify_stage_complete(video_path, stage):
            self.emit_event("stage_skip", video=video_path, stage=stage, gpu=gpu)
            continue
        else:
            # Status says completed but output missing - need to rerun
            self.emit_event(
                "stage_revalidate",
                video=video_path,
                stage=stage,
                gpu=gpu,
                reason="output_missing"
            )
            task.stage_status[stage] = "pending"
```

## Benefits

1. **Robust resume**: Handles cases where outputs are deleted/corrupted
2. **Self-healing**: Automatically detects and regenerates missing outputs
3. **Transparent**: Logs `stage_revalidate` events for debugging
4. **No false skips**: Only skips when output is verified to exist

## Testing

The fix ensures Test 4 in `test_smoke.sh` now passes:
1. Delete `world_space_res.pth`
2. Re-run with `--resume`
3. ✓ System detects missing output
4. ✓ Regenerates the file
5. ✓ Test passes

## Event Log Example

Before fix:
```json
{"event": "stage_skip", "stage": "infiller", "video": "test.mp4"}
// Output missing but skipped anyway - BUG
```

After fix:
```json
{"event": "stage_revalidate", "stage": "infiller", "video": "test.mp4", "reason": "output_missing"}
{"event": "stage_attempt", "stage": "infiller", "video": "test.mp4", "attempt": 0}
{"event": "stage_success", "stage": "infiller", "video": "test.mp4"}
// Output regenerated correctly - FIXED
```

## Backward Compatibility

✓ Fully backward compatible
✓ No changes to command-line interface
✓ No changes to output format
✓ Existing status.json files work correctly
✓ Only adds validation, doesn't remove functionality
