#!/usr/bin/env python3
"""Measure-and-select NCCL_P2P_LEVEL: run real collectives at each candidate
level, verify payload integrity, pick the fastest level that is provably
correct. Companion to derive_nccl_p2p_level.py (which supplies the candidate
set and the safe fallback) and an extension of the SparkInfer #81
calibration-probe philosophy: measured decisions, fingerprint-cached,
explicit values always win.

Selection contract
------------------
1. Candidates: from the static topology derivation — every NCCL level from
   the most conservative up to the literal worst link class (plus NVL when
   NVLink is present). Example on a switched workstation whose worst pair is
   NODE: [PIX, PXB, PHB, SYS].
2. Per candidate, in a fresh killable subprocess group (one rank per GPU),
   with NCCL_P2P_LEVEL exported before communicator init:
     - warmup iterations (excluded from timing);
     - timed all_gather and all_reduce at DCP-traffic-shaped sizes;
     - BIT-VERIFICATION every iteration: all_gather payloads are
       rank-tagged patterns checked element-exactly; all_reduce of a known
       ramp is checked against the closed-form sum. A level that corrupts
       is FAILED regardless of speed.
     - a hard wall-clock timeout enforced by the parent: a hung trial
       (the classic broken-P2P wedge) is killed and marked FAILED.
3. Winner: fastest verified level by the primary metric (all_gather busbw at
   the largest size); ties within 2% resolve to the more conservative level.
4. Result cached in a JSON keyed by fingerprint (topology matrix hash, GPU
   name, driver, PCIe gen/width, world size, probe revision, payload sizes).
   Cache hit => no GPU work. --refresh forces remeasurement.
5. Exit 0 prints the selected level on stdout (everything else on stderr).
   Any failure of the probe machinery itself exits nonzero so the launcher
   falls back to the static derivation. Levels that FAIL verification are a
   normal result (they are excluded), not a probe failure.

Launcher integration:

    if [ "${NCCL_P2P_LEVEL:-auto}" = "auto" ]; then
        LEVEL=$(python3 nccl_p2p_probe.py --devices "$CUDA_VISIBLE_DEVICES") \
          || LEVEL=$(python3 derive_nccl_p2p_level.py --devices "$CUDA_VISIBLE_DEVICES") \
          || LEVEL=""
        [ -n "$LEVEL" ] && export NCCL_P2P_LEVEL="$LEVEL"
    fi

`--plan` prints the trial plan without touching a GPU (CI-testable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

PROBE_REVISION = "nccl-p2p-probe-v1"
# Ordered most-permissive last. NVL is only a candidate when NVLink exists.
_LEVEL_ORDER = ["NVL", "PIX", "PXB", "PHB", "SYS"]
# DCP-traffic-shaped payloads: candidate-exchange scale and CKV-gather scale.
_SIZES_MB = [4, 64]
_WARMUP = 5
_ITERS = 20
_TRIAL_TIMEOUT_S = 60
_TIE_BAND = 0.02


def _topo_text() -> str:
    return subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout


def _candidates_and_fallback(devices: str) -> tuple[list[str], str]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import derive_nccl_p2p_level as st

    pairs = st.parse_topo_matrix(_topo_text())
    devs = st._selected_devices(devices or None, pairs)
    if len(devs) < 2:
        raise SystemExit(2)
    capped, worst = st.derive_level(pairs, devs, permissive=False)
    literal, _ = st.derive_level(pairs, devs, permissive=True)
    lo = _LEVEL_ORDER.index(capped if capped != "NVL" else "NVL")
    hi = _LEVEL_ORDER.index(literal)
    cands = _LEVEL_ORDER[min(lo, hi): max(lo, hi) + 1]
    if "NVL" in (capped, literal) and "NVL" not in cands:
        cands = ["NVL"] + cands
    print(
        f"nccl_p2p_probe: worst-pair={worst[0]}-{worst[1]}({worst[2]}) "
        f"static-fallback={capped} candidates={cands}",
        file=sys.stderr,
    )
    return cands, capped


def _fingerprint(devices: str, world: int) -> str:
    q = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    blob = "|".join([
        PROBE_REVISION, devices or "all", str(world), q.strip(),
        hashlib.sha256(_topo_text().encode()).hexdigest(),
        json.dumps(_SIZES_MB),
    ])
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


# ----------------------------------------------------------------- worker --
_WORKER = r"""
import json, os, sys, time
import torch, torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world)
dev = torch.device("cuda", rank)
sizes = json.loads(os.environ["PROBE_SIZES_MB"])
warmup = int(os.environ["PROBE_WARMUP"]); iters = int(os.environ["PROBE_ITERS"])
out = {}
for mb in sizes:
    n = mb * 1024 * 1024 // 4
    # Rank-tagged pattern: exact bit-verification after all_gather.
    src = (torch.arange(n, device=dev, dtype=torch.float32) % 977) + rank * 1000.0
    gat = [torch.empty(n, device=dev, dtype=torch.float32) for _ in range(world)]
    ramp = torch.full((n,), 1.0 + rank, device=dev, dtype=torch.float32)
    expect_red = float(sum(range(1, world + 1)))
    base = (torch.arange(n, device=dev, dtype=torch.float32) % 977)
    for it in range(warmup + iters):
        if it == warmup:
            torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
        dist.all_gather(gat, src)
        red = ramp.clone(); dist.all_reduce(red)
    torch.cuda.synchronize(); dist.barrier()
    dt = (time.perf_counter() - t0) / iters
    ok = all(torch.equal(gat[r], base + r * 1000.0) for r in range(world))
    ok = ok and bool(torch.all(red == expect_red).item())
    # all_gather busbw (NCCL convention): bytes*(world-1)/world / time.
    busbw = (n * 4 * world * (world - 1) / world) / dt / 1e9
    out[str(mb)] = {"verified": ok, "ag_busbw_GBs": round(busbw, 3),
                    "step_ms": round(dt * 1e3, 3)}
dist.destroy_process_group()
if rank == 0:
    print(json.dumps(out))
sys.stdout.flush()
"""


def _run_trial(level: str, devices: str, world: int) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_WORKER)
        worker = f.name
    env = dict(os.environ)
    env.update({
        "NCCL_P2P_LEVEL": level,
        "CUDA_VISIBLE_DEVICES": devices or env.get("CUDA_VISIBLE_DEVICES", ""),
        "PROBE_SIZES_MB": json.dumps(_SIZES_MB),
        "PROBE_WARMUP": str(_WARMUP),
        "PROBE_ITERS": str(_ITERS),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(29400 + os.getpid() % 500),
        # Fail rather than silently fall back inside NCCL where possible.
        "NCCL_DEBUG": env.get("NCCL_DEBUG", "WARN"),
    })
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--standalone", f"--nproc-per-node={world}", worker]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=_TRIAL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"status": "FAILED", "reason": f"timeout>{_TRIAL_TIMEOUT_S}s (hang)"}
    finally:
        os.unlink(worker)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return {"status": "FAILED", "reason": "nonzero exit", "stderr_tail": tail}
    try:
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
        res = json.loads(line)
    except Exception:
        return {"status": "FAILED", "reason": "unparseable worker output"}
    if not all(v["verified"] for v in res.values()):
        return {"status": "FAILED", "reason": "verification mismatch", "data": res}
    res["status"] = "OK"
    res["elapsed_s"] = round(time.monotonic() - t0, 1)
    return res


def _select(results: dict[str, dict]) -> str | None:
    big = str(max(_SIZES_MB))
    ok = {lv: r for lv, r in results.items() if r.get("status") == "OK"}
    if not ok:
        return None
    best_bw = max(r[big]["ag_busbw_GBs"] for r in ok.values())
    within = [lv for lv, r in ok.items()
              if r[big]["ag_busbw_GBs"] >= best_bw * (1 - _TIE_BAND)]
    # Conservative tiebreak: earliest in _LEVEL_ORDER wins.
    within.sort(key=_LEVEL_ORDER.index)
    return within[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    ap.add_argument("--cache-dir", default="/root/.cache/nccl-p2p-probe")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="print candidates and trial plan; no GPU work")
    ap.add_argument("--total-budget-s", type=float, default=180.0,
                    help="hard wall-clock budget for all trials combined")
    ap.add_argument("--sizes-mb", default=None,
                    help="comma-separated payload MB (default 4,64); smaller "
                         "sizes allow probing beside a memory-resident server")
    args = ap.parse_args()
    if args.sizes_mb:
        global _SIZES_MB
        _SIZES_MB = [int(x) for x in args.sizes_mb.split(",") if x.strip()]

    # An explicitly configured level ALWAYS wins: echo it, run nothing.
    # This holds even if a launcher mis-wires the auto gate.
    explicit = os.environ.get("NCCL_P2P_LEVEL", "").strip()
    if explicit and explicit.lower() != "auto":
        print(
            f"nccl_p2p_probe: NCCL_P2P_LEVEL={explicit} set explicitly; "
            "respecting it, no probe", file=sys.stderr,
        )
        print(explicit)
        return 0

    try:
        cands, fallback = _candidates_and_fallback(args.devices)
    except SystemExit as e:
        return int(e.code or 2)
    except Exception as exc:  # noqa: BLE001 - launcher falls back to static
        print(f"nccl_p2p_probe: candidate derivation failed: {exc}", file=sys.stderr)
        return 3

    world = len((args.devices or "").split(",")) if args.devices else None
    if world is None:
        import re as _re
        world = len(_re.findall(r"^GPU\d+", _topo_text(), _re.M))

    if args.plan:
        print(f"plan: candidates={cands} world={world} sizes_MB={_SIZES_MB} "
              f"warmup={_WARMUP} iters={_ITERS} timeout={_TRIAL_TIMEOUT_S}s "
              f"fallback={fallback}", file=sys.stderr)
        print(fallback)
        return 0

    fp = _fingerprint(args.devices, world)
    cache = os.path.join(args.cache_dir, f"{fp}.json")
    if not args.refresh and os.path.exists(cache):
        with open(cache) as f:
            saved = json.load(f)
        print(f"nccl_p2p_probe: cache hit {fp} -> {saved['selected']}",
              file=sys.stderr)
        print(saved["selected"])
        return 0

    results: dict[str, dict] = {}
    budget_t0 = time.monotonic()
    for lv in cands:
        spent = time.monotonic() - budget_t0
        if spent > args.total_budget_s:
            print(
                f"nccl_p2p_probe: total budget {args.total_budget_s}s exceeded "
                f"({spent:.0f}s); skipping remaining candidates", file=sys.stderr,
            )
            break
        print(f"nccl_p2p_probe: trial NCCL_P2P_LEVEL={lv} ...", file=sys.stderr)
        results[lv] = _run_trial(lv, args.devices, world)
        print(f"nccl_p2p_probe:   {lv}: {json.dumps(results[lv])[:200]}",
              file=sys.stderr)

    selected = _select(results)
    if selected is None:
        print("nccl_p2p_probe: no candidate verified; use static fallback",
              file=sys.stderr)
        return 4

    os.makedirs(args.cache_dir, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"fingerprint": fp, "selected": selected,
                   "results": results, "candidates": cands,
                   "revision": PROBE_REVISION}, f, indent=1)
    print(f"nccl_p2p_probe: selected {selected} (cached {fp})", file=sys.stderr)
    print(selected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
