# LEADER inference profiling

Per-stage profiling of the LEADER PyTorch pipeline on a single GPU. Measured
with [`pyfing/pytorch/tools/profile.py`](../tools/profile.py), which times each
stage independently using CUDA events (GPU stages) and `time.perf_counter`
(CPU stages):

- `data_wait` — wall-clock waiting for the DataLoader to yield the next batch
  (small if the 4 worker processes keep up; larger if I/O-bound)
- `h2d` — host-to-device transfer (`.to(device, non_blocking=True)`)
- `forward` — model forward pass (`LeaderNet`)
- `d2h` — device-to-host of the post-processed minutiae arrays (single batched
  transfer per step)
- `postproc` — threshold + sort per image; runs on GPU (default `--postproc torch`)
  or numpy CPU (legacy `--postproc numpy`)

File saving is intentionally excluded to isolate the cost of inference itself.

## Setup

| | |
|---|---|
| GPU | NVIDIA H100 PCIe (id=0), 80 GB |
| Host | grh100 |
| PyTorch | 2.10.0+cu128 (CUDA 12.8, cuDNN 9.10.2) |
| Python | 3.12.11 |
| Precision | TF32 (matmul + cuDNN conv) |
| Batch size | 8 |
| DataLoader workers | 4, `pin_memory=True`, `persistent_workers=True` |
| Strategy | full GPU forward; CPU post-processing in profile loop |
| Warmup | 10 batches before measurement |
| Date | 2026-05-05 |

`torch.compile` is **off**. `cudnn.benchmark` is **off** (default heuristics).

## Datasets

| Dataset | Images | Source size | Padded H×W (mode) |
|---|---:|---|---|
| SD258 (full) | 516 | 800×768 | 768×800 |
| BN48k (first 1000, sorted) | 1000 | 512×512 | 512×512 |

Latents in SD258 are noticeably larger than the rolled fingerprints in BN48k —
the H100 sees roughly 2.4× more pixels per image, which dominates the
inference-time difference reported below.

## Throughput summary

Two throughput numbers are reported:

- **Steady state**: per-stage times summed over the modal-shape batches,
  excluding the first profiled batch and any partial trailing batch. Represents
  the cost of inference once cuDNN has set up workspaces for the operating shape.
- **Wall (incl. cold start)**: total wall-clock divided by all images.
  Lower than steady state because cuDNN's first call with a new tensor shape
  carries a one-off setup cost — see *Cold-start cost* below.

| Dataset | Wall throughput | Steady-state throughput | ms/img (steady) |
|---|---:|---:|---:|
| SD258 (516 imgs) | 33.82 imgs/s | **59.49 imgs/s** | 16.81 ms |
| BN48k-1k (1000 imgs) | 128.97 imgs/s | **133.53 imgs/s** | 7.49 ms |

The wall figure matters when you run on a heterogeneous mix of shapes;
steady-state is the right number for a same-shape stream.

> **GPU post-processing optimization (2026-05-05).** The numbers above use the
> new GPU-resident extractor (`_extract_minutiae_torch`), which threshold-and-
> sorts on the device and transfers only the small `[N, 6]` minutiae arrays.
> The previous numpy path D2H'd the full `[B, 4, H, W]` feature map per batch.
> Verified equivalent on a 32-image SD258+BN48k sample at thresholds 0.6 and
> 0.15: identical minutiae set in 100% of cases; identical line ordering in
> 30/32 (the remaining 2 differ only when two minutiae share a quality score
> exactly — an extremely rare tie). See [Comparison with FingerNet](#comparison-with-fingernet-on-the-same-machine).

## Per-stage breakdown — SD258 (768×800 padded, batch=8, GPU postproc)

```
stage          mean     median       p95    mean/img    med/img
data_wait     0.21ms     0.18ms     0.44ms    0.027ms    0.022ms
h2d           0.78ms     0.76ms     0.90ms    0.097ms    0.095ms
forward     130.52ms   130.52ms   130.54ms   16.315ms   16.315ms
d2h           0.10ms     0.10ms     0.12ms    0.013ms    0.012ms
postproc      3.08ms     2.76ms     4.43ms    0.385ms    0.345ms
```

- **Forward now dominates fully** (~97% of per-batch wall) — d2h and postproc
  are both <1% of the time.
- **D2H collapsed from 50.86 ms to 0.10 ms** (500× reduction): we transfer only
  the per-image minutiae arrays (~30,000 floats total per batch ≈ 120 KB)
  instead of the 33 MB feature map.
- **Postproc dropped from 13.78 ms to 3.08 ms** (4.5×) — `torch.nonzero` +
  `torch.argsort` on GPU vs numpy on the host.
- 30,720 minutiae extracted (mean 59.5/image at threshold = 0.6).

## Per-stage breakdown — BN48k (512×512 padded, batch=8, GPU postproc)

```
stage          mean     median       p95    mean/img    med/img
data_wait     0.24ms     0.19ms     0.42ms    0.030ms    0.024ms
h2d           0.36ms     0.35ms     0.44ms    0.046ms    0.044ms
forward      56.39ms    56.36ms    56.50ms    7.049ms    7.044ms
d2h           0.11ms     0.10ms     0.14ms    0.013ms    0.012ms
postproc      3.05ms     2.76ms     3.94ms    0.381ms    0.345ms
```

- D2H is now negligible regardless of image size — the transferred minutiae
  arrays scale with the number of detected minutiae, not the feature map size.
- 61,643 minutiae extracted (mean 61.6/image at threshold = 0.6).

## Per-image cost (steady-state, ms/image)

| Stage | SD258 (768×800) | BN48k (512×512) |
|---|---:|---:|
| data_wait | 0.03 | 0.03 |
| h2d | 0.10 | 0.05 |
| forward | **16.32** | **7.05** |
| d2h | 0.01 | 0.01 |
| postproc | 0.39 | 0.38 |
| **sum** | **16.85** | **7.52** |

Forward is now ~97% of inference time — there is essentially nothing else to
optimize without touching the model itself. The sum matches the steady-state
throughput (1000/16.8 ≈ 59, 1000/7.5 ≈ 133).

### Before/after — what the GPU postproc bought us

| Dataset | Stage | Legacy (numpy) | Optimized (torch) | Speedup |
|---|---|---:|---:|---:|
| SD258 | d2h (ms/batch) | 50.86 | 0.10 | **509×** |
| SD258 | postproc (ms/batch) | 13.78 | 3.08 | 4.5× |
| SD258 | **steady throughput** | **40.79 imgs/s** | **59.49 imgs/s** | **+46%** |
| BN48k | d2h (ms/batch) | 24.51 | 0.11 | 223× |
| BN48k | postproc (ms/batch) | 7.65 | 3.05 | 2.5× |
| BN48k | **steady throughput** | **89.76 imgs/s** | **133.53 imgs/s** | **+49%** |

After this change, LEADER's BN48k throughput (133.5 imgs/s) **exceeds the
optimized FingerNet's** (112.9 imgs/s) — see comparison section below.

## Cold-start cost (first call per shape)

cuDNN does lazy workspace setup the first time it sees a new tensor shape —
even with the default heuristic algorithm, not just with `cudnn.benchmark=True`.
This appears as a one-off latency spike on:

1. The very first profiled batch (mostly absorbed by the 10-batch warmup).
2. Any partial trailing batch whose `B` differs from the steady-state `B`.

Observed in this run:

| Run | Excluded batches (forward time) |
|---|---|
| SD258 | 131 ms (first), **9289 ms** (last batch, B=4 vs B=8) |
| BN48k | 57 ms (first); no partial batch |

For SD258, the partial last batch costs more than the entire rest of the
dataset combined (9.3s vs ~8.4s of steady forward across 64 full batches).

**Practical implication**: if you batch a heterogeneous workload, group images
by shape — every distinct (B, H, W) shape pays a one-off setup of several
seconds. The dynamic-padding collate already aligns H/W to multiples of 32,
which keeps the unique-shape count low for uniform datasets.

## Comparison with FingerNet on the same machine

After the GPU post-processing optimization, **LEADER beats FingerNet on
BN48k**: 133.5 imgs/s vs 112.9 imgs/s on the same H100 PCIe (batch 8). The
section below preserves the pre-optimization analysis that diagnosed the gap;
the headline numbers in this paragraph are the *current* state.

The pre-optimization gap was 35% (FingerNet 112.9 vs LEADER 83.8 imgs/s) and
came entirely from **D2H + post-processing**, not from the forward pass:
LEADER and FingerNet have nearly identical forward times.

### Parameters

| Model | Parameters | FP32 size |
|---|---:|---:|
| LEADER | 948,114 | 3.8 MB |
| FingerNet | 4,709,694 | 18.8 MB |

LEADER has 5× fewer parameters than FingerNet. (Counted with
`sum(p.numel() for p in model.parameters())` on the PyTorch ports of both
models.)

### Forward-only time (apples-to-apples, same GPU, same batch, same input)

Measured by feeding the same `[8, 1, H, W]` tensor 50 times to each model on
GPU 0 (TF32 ON, batch=8, 15-batch warmup, drop first measured iter):

**Without `cudnn.benchmark`** (default — what we use in production for
byte-level reproducibility):

| Dataset | LEADER | FingerNet | Winner |
|---|---:|---:|---|
| BN48k 512×512 | **7.05 ms/img** | 8.87 ms/img | **LEADER, 21% faster** |
| SD258 768×800 | **16.28 ms/img** | 21.01 ms/img | **LEADER, 23% faster** |

**With `cudnn.benchmark=True`** (peak-throughput mode — what the FingerNet
profile uses):

| Dataset | LEADER | FingerNet | Winner |
|---|---:|---:|---|
| BN48k 512×512 | 6.64 ms/img | **6.23 ms/img** | FingerNet, 7% |
| SD258 768×800 | 15.50 ms/img | **14.77 ms/img** | FingerNet, 5% |

cuDNN autotune helps FingerNet ~30% but LEADER only ~6%: dense 3×3/5×5 convs
have many candidate algorithms to choose from; depthwise 7×7 convs (used in
LEADER's inverted-bottleneck blocks) are bandwidth-limited and don't have
much room for autotune.

### Where the end-to-end gap comes from

| Stage (BN48k, 512×512, batch=8) | FingerNet | LEADER | Δ |
|---|---:|---:|---|
| Forward | 6.25 ms/img | 7.05 ms/img | +13% |
| H2D | 0.06 ms/img | 0.06 ms/img | — |
| D2H | 0.25 ms/img | **3.61 ms/img** | **+14×** |
| Post-processing | 1.98 ms/img *(on GPU)* | 1.12 ms/img *(on CPU)* | — |
| Total | 8.86 ms/img | 11.94 ms/img | +35% |

> The "+13%" forward column is FingerNet-with-benchmark vs LEADER-without —
> the comparable values from each project's published profile. With both at
> the same setting the forwards are within 5–7% (see table above).

The dominant factor is the **D2H of the full-resolution 4-channel feature map**.
LEADER's `_run_full_gpu` does `raw.permute(0,2,3,1).cpu().numpy()` on the
entire `[B, 4, H, W]` output ([api.py:509](../api.py#L509)). For BN48k that's
33.5 MB / batch over PCIe (8 × 4 × 512 × 512 × 4B), measured at **3.6 ms/img**.

FingerNet runs NMS, minutiae detection, orientation field, and enhanced image
generation on the GPU — only the small `suppress_mask` and final minutiae
list cross PCIe (**0.25 ms/img**, ~14× cheaper).

This is also how the original Keras code does it (`pyfing.minutiae.Leader`):
`self.model(...).numpy()[0]` followed by `np.argwhere(out[...,3] >= τ)` —
same full-feature-map D2H, same numpy threshold. The bottleneck is in
LEADER's pipeline design, not in the PyTorch port.

### Summary — applied optimization (2026-05-05)

The single change with the biggest expected impact was **moving post-processing
to the GPU**: replace the full feature-map D2H + numpy NMS with a GPU
`torch.nonzero` on `out[..., 3] >= τ` and transfer only the per-image
minutiae arrays. Implemented in
[`_extract_minutiae_torch`](../api.py#L210) and used by
[`_run_full_gpu`](../api.py#L487); `_run_hybrid` keeps the numpy path
because that strategy intentionally offloads to CPU workers.

Measured impact (BN48k 512×512, batch 8):

| Stage | Before (numpy) | After (torch) | Speedup |
|---|---:|---:|---:|
| D2H | 3.06 ms/img | 0.013 ms/img | 235× |
| Postproc | 0.96 ms/img | 0.38 ms/img | 2.5× |
| Forward | 7.05 ms/img | 7.05 ms/img | unchanged |
| **Steady throughput** | **89.76 imgs/s** | **133.53 imgs/s** | **+49%** |

Output equivalence verified on a 32-image SD258+BN48k sample at thresholds
0.6 and 0.15: identical minutiae set in 100% of cases (3922 == 3922 minutiae
at threshold 0.15; 2247 == 2247 at threshold 0.6). 30/32 files are byte-
identical to the numpy output; 2/32 differ only in the order of two equal-
quality minutiae (`torch.argsort(stable=True)` breaks float32 quality ties
differently from numpy's quicksort, but the values themselves match).

### Further gains (not applied, riskier)

After the GPU post-processing optimization, forward time is ~97% of the
inference cost. Remaining optimization options carry numeric drift risk:

- **`cudnn.benchmark=True`**: ~6% on LEADER's forward (vs ~30% on FingerNet,
  because LEADER's depthwise 7×7 convs have less algorithm choice for
  autotune). Off by default for byte-level reproducibility.
- **`torch.compile`** and **channels-last** could push another 10–20%
  but may shift outputs at the rounding boundary (see
  [PROFILE fingernet.md](PROFILE%20fingernet.md) Reproducibility section).

## How to reproduce

```bash
# SD258 (full)
python -m pyfing.pytorch.tools.profile \
    --input /storage/jcontreras/data/datasets/fingerprints/SD258/images/ \
    --label SD258 --gpu 0 --batch-size 8 --warmup 10 \
    --report-json /tmp/profile_sd258.json

# BN48k (first 1000, sorted)
find /storage/jcontreras/data/datasets/fingerprints/BN48k/images/ -name '*.png' \
    | sort | head -1000 > /tmp/bn48k_first1000.list

python -m pyfing.pytorch.tools.profile \
    --input /tmp/bn48k_first1000.list \
    --label BN48k-first1000 --gpu 0 --batch-size 8 --warmup 10 \
    --report-json /tmp/profile_bn48k.json
```

To compare with the legacy numpy post-processing path, add `--postproc numpy`.
