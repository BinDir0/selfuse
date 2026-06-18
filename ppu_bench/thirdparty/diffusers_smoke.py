"""Real codebase: diffusers (EgoVLA pins diffusers==0.34.0).

EgoVLA's flow/diffusion action head depends on diffusers. This runs a real
diffusers UNet + scheduler denoising loop on the PPU - no big weight download,
just the library's own ops/scheduler math - and checks outputs stay finite.

    python diffusers_smoke.py
"""

from __future__ import annotations

import torch

from _bench import DEVICE, IS_ACCEL, fail, ok, section, skip, timed


def run() -> None:
    section("diffusers UNet + scheduler denoise loop")
    if not IS_ACCEL:
        skip("no accelerator")
        return
    try:
        from diffusers import DDPMScheduler, UNet2DModel
    except Exception as exc:  # noqa: BLE001
        skip(f"diffusers unavailable: {type(exc).__name__}: {exc}")
        return

    try:
        model = UNet2DModel(
            sample_size=32, in_channels=4, out_channels=4,
            layers_per_block=2, block_out_channels=(64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        ).to(DEVICE).to(torch.bfloat16).eval()
        sched = DDPMScheduler(num_train_timesteps=1000)
        sched.set_timesteps(25)

        x = torch.randn(2, 4, 32, 32, device=DEVICE, dtype=torch.bfloat16)

        def denoise():
            s = x.clone()
            with torch.no_grad():
                for t in sched.timesteps:
                    noise = model(s, t).sample
                    s = sched.step(noise, t, s).prev_sample
            return s

        out = denoise()
        finite = bool(torch.isfinite(out).all())
        secs = timed(denoise, iters=3, warmup=1)
        if finite:
            ok(f"25-step denoise ok, finite, {secs * 1e3:.0f} ms  out={tuple(out.shape)}")
        else:
            fail("denoise produced non-finite values")
    except Exception as exc:  # noqa: BLE001
        fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run()
