"""Per-stage profiling of LEADER inference on a single GPU.

Measures, per batch, the wall-clock cost of each pipeline stage:
  - data_wait:   time waited for the DataLoader to yield the next batch
                 (small if workers keep up; large if I/O bound)
  - h2d:         host-to-device transfer (CUDA event)
  - forward:     model forward pass (CUDA event)
  - d2h:         device-to-host transfer + permute (CUDA event)
  - postproc:    NMS threshold + sort to produce minutiae arrays (CPU)

Saving is deliberately skipped to keep the measurement focused on inference.
Aggregates per-stage mean/median/p95 in milliseconds and reports overall
throughput (images/second) for the chosen dataset.

Usage (from repo root):
    python -m pyfing.pytorch.tools.profile \\
        --input /path/to/images --batch-size 8 --gpu 0 \\
        [--limit N] [--label NAME] [--report-json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..api import (
    FingerprintDataset,
    _extract_minutiae_numpy,
    _extract_minutiae_torch,
    dynamic_padding_collate,
    find_image_paths,
)
from ..leader_model import LeaderNet
from ..registry import get_model_spec

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"


@dataclass
class StageTimings:
    data_wait_ms: list[float] = field(default_factory=list)
    h2d_ms: list[float] = field(default_factory=list)
    forward_ms: list[float] = field(default_factory=list)
    d2h_ms: list[float] = field(default_factory=list)
    postproc_ms: list[float] = field(default_factory=list)
    batch_size: list[int] = field(default_factory=list)
    batch_h: list[int] = field(default_factory=list)
    batch_w: list[int] = field(default_factory=list)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summary(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def profile(
    image_paths: list[str],
    *,
    label: str,
    gpu_id: int,
    batch_size: int,
    num_workers: int,
    threshold: float,
    type_threshold: float,
    warmup_batches: int,
    dpi: int,
    max_image_dim: int,
    postproc: str = "torch",   # "torch" (GPU extraction, default) or "numpy" (legacy)
) -> dict:
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    spec = get_model_spec("leader")
    state = torch.load(str(spec.torch_weights), map_location=device, weights_only=True)
    model = LeaderNet().to(device)
    model.load_state_dict(state)
    model.eval()

    dataset = FingerprintDataset(image_paths, dpi=dpi, max_dim=max_image_dim)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=dynamic_padding_collate,
    )

    timings = StageTimings()
    n_images_processed = 0
    n_minutiae_total = 0

    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    d2h_start = torch.cuda.Event(enable_timing=True)
    d2h_end = torch.cuda.Event(enable_timing=True)

    print(f"[{label}] warmup {warmup_batches} batches…", flush=True)
    warmup_done = 0
    with torch.no_grad():
        for batch_tensors, _, _ in loader:
            if batch_tensors is None:
                continue
            # Warm CUDA events too (avoids first-event-record latency in profile loop)
            h2d_start.record()
            x = batch_tensors.to(device, non_blocking=True)
            h2d_end.record()
            fwd_start.record()
            _ = model(x)
            fwd_end.record()
            torch.cuda.synchronize()
            warmup_done += 1
            if warmup_done >= warmup_batches:
                break

    print(f"[{label}] profiling {len(image_paths)} images, batch_size={batch_size}…", flush=True)
    wall_start = time.perf_counter()
    last_iter_end = time.perf_counter()

    with torch.no_grad():
        for batch_tensors, batch_paths, batch_orig_shapes in loader:
            data_wait_ms = (time.perf_counter() - last_iter_end) * 1000.0

            if batch_tensors is None:
                last_iter_end = time.perf_counter()
                continue

            B, _, H, W = batch_tensors.shape

            h2d_start.record()
            x = batch_tensors.to(device, non_blocking=True)
            h2d_end.record()

            fwd_start.record()
            raw = model(x)
            fwd_end.record()

            orig_hs, orig_ws = batch_orig_shapes

            if postproc == "torch":
                # GPU extraction: threshold + sort per image on GPU,
                # transfer only the resulting [N_i, 6] minutiae arrays.
                pp_start_event = torch.cuda.Event(enable_timing=True)
                pp_end_event = torch.cuda.Event(enable_timing=True)
                pp_start_event.record()
                gpu_results: list[torch.Tensor] = []
                lengths: list[int] = []
                for i in range(B):
                    oh = orig_hs[i].item()
                    ow = orig_ws[i].item()
                    m_gpu = _extract_minutiae_torch(
                        raw[i, :, :oh, :ow], threshold, type_threshold,
                    )
                    gpu_results.append(m_gpu)
                    lengths.append(int(m_gpu.shape[0]))
                pp_end_event.record()

                d2h_start.record()
                if any(l > 0 for l in lengths):
                    packed = torch.cat(gpu_results, dim=0).cpu().numpy()
                else:
                    packed = np.empty((0, 6), dtype=np.float32)
                d2h_end.record()

                torch.cuda.synchronize()
                pp_ms = pp_start_event.elapsed_time(pp_end_event)
                n_minutiae_total += sum(lengths)
            else:
                # Legacy numpy path: D2H of full feature map, threshold + sort in numpy.
                d2h_start.record()
                out_nhwc = raw.permute(0, 2, 3, 1).cpu().numpy()
                d2h_end.record()

                torch.cuda.synchronize()

                pp_t0 = time.perf_counter()
                for i in range(B):
                    oh = orig_hs[i].item()
                    ow = orig_ws[i].item()
                    m = _extract_minutiae_numpy(
                        out_nhwc[i, :oh, :ow, :], threshold, type_threshold,
                    )
                    n_minutiae_total += int(m.shape[0])
                pp_ms = (time.perf_counter() - pp_t0) * 1000.0

            h2d_ms = h2d_start.elapsed_time(h2d_end)
            fwd_ms = fwd_start.elapsed_time(fwd_end)
            d2h_ms = d2h_start.elapsed_time(d2h_end)

            timings.data_wait_ms.append(data_wait_ms)
            timings.h2d_ms.append(h2d_ms)
            timings.forward_ms.append(fwd_ms)
            timings.d2h_ms.append(d2h_ms)
            timings.postproc_ms.append(pp_ms)
            timings.batch_size.append(B)
            timings.batch_h.append(H)
            timings.batch_w.append(W)
            n_images_processed += B

            last_iter_end = time.perf_counter()

    wall_total_s = time.perf_counter() - wall_start
    throughput = n_images_processed / wall_total_s if wall_total_s > 0 else 0.0

    # Steady-state stats: keep only batches whose tensor shape (B, H, W) matches
    # the modal shape, AND drop the first profiled batch. This removes:
    #   - one-off cuDNN/CUDA workspace setup on the first profiled forward
    #   - the trailing partial batch (different B), which triggers a fresh
    #     cuDNN setup costing several seconds on H100 for large shapes
    modal_b = statistics.mode(timings.batch_size)
    modal_h = statistics.mode(timings.batch_h)
    modal_w = statistics.mode(timings.batch_w)
    keep = [
        i for i in range(len(timings.forward_ms))
        if i > 0
        and timings.batch_size[i] == modal_b
        and timings.batch_h[i] == modal_h
        and timings.batch_w[i] == modal_w
    ]

    def _ss(values: list[float]) -> list[float]:
        return [values[i] for i in keep]

    ss_imgs = sum(timings.batch_size[i] for i in keep)
    ss_wall_s = sum(
        timings.h2d_ms[i] + timings.forward_ms[i]
        + timings.d2h_ms[i] + timings.postproc_ms[i]
        for i in keep
    ) / 1000.0
    ss_throughput = ss_imgs / ss_wall_s if ss_wall_s > 0 else 0.0

    per_image = {
        "data_wait_ms": [v / b for v, b in zip(timings.data_wait_ms, timings.batch_size)],
        "h2d_ms": [v / b for v, b in zip(timings.h2d_ms, timings.batch_size)],
        "forward_ms": [v / b for v, b in zip(timings.forward_ms, timings.batch_size)],
        "d2h_ms": [v / b for v, b in zip(timings.d2h_ms, timings.batch_size)],
        "postproc_ms": [v / b for v, b in zip(timings.postproc_ms, timings.batch_size)],
    }

    result = {
        "label": label,
        "n_images": n_images_processed,
        "n_batches": len(timings.forward_ms),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "gpu": torch.cuda.get_device_name(gpu_id),
        "gpu_id": gpu_id,
        "wall_time_s": wall_total_s,
        "throughput_imgs_per_s": throughput,
        "steady_state_throughput_imgs_per_s": ss_throughput,
        "steady_state_batches": len(keep),
        "first_batch_forward_ms": timings.forward_ms[0] if timings.forward_ms else None,
        "excluded_batches_forward_ms": [
            timings.forward_ms[i]
            for i in range(len(timings.forward_ms)) if i not in keep
        ],
        "minutiae_total": n_minutiae_total,
        "minutiae_per_image_mean": n_minutiae_total / n_images_processed if n_images_processed else 0.0,
        "padded_hw_mode": (
            statistics.mode(timings.batch_h) if timings.batch_h else None,
            statistics.mode(timings.batch_w) if timings.batch_w else None,
        ),
        "per_batch_ms": {
            "data_wait": _summary(_ss(timings.data_wait_ms)),
            "h2d": _summary(_ss(timings.h2d_ms)),
            "forward": _summary(_ss(timings.forward_ms)),
            "d2h": _summary(_ss(timings.d2h_ms)),
            "postproc": _summary(_ss(timings.postproc_ms)),
        },
        "per_image_ms": {k: _summary(_ss(v)) for k, v in per_image.items()},
    }

    return result


def _print_report(r: dict):
    pb = r["per_batch_ms"]
    pi = r["per_image_ms"]
    print()
    print(f"=== {r['label']} ===")
    print(f"images:     {r['n_images']}  batches: {r['n_batches']}  batch_size: {r['batch_size']}")
    print(f"gpu:        {r['gpu']} (id={r['gpu_id']})")
    print(f"padded HW:  {r['padded_hw_mode'][0]}x{r['padded_hw_mode'][1]} (mode)")
    print(f"wall:       {r['wall_time_s']:.2f} s")
    print(f"throughput: {r['throughput_imgs_per_s']:.2f} imgs/s (incl. cold start)")
    excluded = r['excluded_batches_forward_ms']
    excluded_str = ", ".join(f"{v:.0f}ms" for v in excluded) if excluded else "none"
    print(f"  steady:   {r['steady_state_throughput_imgs_per_s']:.2f} imgs/s "
          f"({r['steady_state_batches']} batches, excluded: {excluded_str})")
    print(f"minutiae:   total={r['minutiae_total']}  mean/img={r['minutiae_per_image_mean']:.1f}")
    print()
    print(f"  Per-stage stats (steady-state):")
    print(f"  {'stage':<10}  {'mean':>9}  {'median':>9}  {'p95':>9}  {'mean/img':>9}  {'med/img':>9}")
    for stage in ("data_wait", "h2d", "forward", "d2h", "postproc"):
        b = pb[stage]
        i = pi[f"{stage}_ms"]
        print(f"  {stage:<10}  {b['mean']:>7.2f}ms  {b['median']:>7.2f}ms  "
              f"{b['p95']:>7.2f}ms  {i['mean']:>7.3f}ms  {i['median']:>7.3f}ms")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Image dir or .txt/.list")
    p.add_argument("--label", default=None, help="Label for this run (default: input basename)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", "-b", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--type-threshold", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--max-dim", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None, help="Profile only the first N images")
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--postproc", choices=["torch", "numpy"], default="torch",
                   help="Post-processing path: 'torch' (GPU, default) or 'numpy' (legacy)")
    p.add_argument("--report-json", default=None)
    args = p.parse_args()

    paths = find_image_paths(args.input, recursive=args.recursive)
    if args.limit:
        paths = paths[: args.limit]

    label = args.label or args.input.rstrip("/").split("/")[-1]
    result = profile(
        paths,
        label=label,
        gpu_id=args.gpu,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        type_threshold=args.type_threshold,
        warmup_batches=args.warmup,
        dpi=args.dpi,
        max_image_dim=args.max_dim,
        postproc=args.postproc,
    )
    _print_report(result)
    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote report: {args.report_json}")


if __name__ == "__main__":
    main()
