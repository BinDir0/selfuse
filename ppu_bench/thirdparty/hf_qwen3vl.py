"""Real codebase: HuggingFace transformers on a Qwen VL model.

This exercises EgoVLA's actual backbone family (Qwen3-VL-2B) end to end on the
PPU: the real model class, its vision tower, and its attention path. We run a
true vision+text generate and check the output is coherent (non-empty, finite)
under both sdpa and flash_attention_2, then time them.

NOTE on versions: EgoVLA uses Qwen3-VL (needs transformers ~5.x). PPU SDK ships
transformers 4.57.x, which may only have Qwen2.5-VL. Pass --model to match what
your installed transformers supports; the default is the safe Qwen2.5-VL.

    python hf_qwen3vl.py --model Qwen/Qwen2.5-VL-3B-Instruct
    python hf_qwen3vl.py --model Qwen/Qwen3-VL-2B-Instruct   # if transformers 5.x
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image, ImageDraw

from _bench import DEVICE, IS_ACCEL, fail, ok, section, skip, timed


def _synthetic_image() -> Image.Image:
    img = Image.new("RGB", (448, 448), (30, 30, 60))
    d = ImageDraw.Draw(img)
    d.rectangle([80, 80, 240, 240], fill=(220, 60, 60))
    d.ellipse([240, 240, 380, 380], fill=(60, 200, 90))
    return img


def run(model_id: str) -> None:
    section(f"HF transformers VL: {model_id}")
    if not IS_ACCEL:
        skip("no accelerator")
        return
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:  # noqa: BLE001
        skip(f"transformers VL classes unavailable: {type(exc).__name__}: {exc}")
        return

    image = _synthetic_image()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What shapes and colors are in this image?"},
        ],
    }]

    for attn in ("sdpa", "flash_attention_2"):
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, attn_implementation=attn
            ).to(DEVICE).eval()
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(DEVICE)

            def gen():
                with torch.no_grad():
                    return model.generate(**inputs, max_new_tokens=48, do_sample=False)

            out_ids = gen()
            text = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[0].strip()
            secs = timed(gen, iters=3, warmup=1)
            if text:
                ok(f"[{attn}] {secs * 1e3:.0f} ms/gen  -> {text[:80]!r}")
            else:
                fail(f"[{attn}] generated empty output")
            del model
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            fail(f"[{attn}] {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    run(ap.parse_args().model)


if __name__ == "__main__":
    main()
