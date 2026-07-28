#!/usr/bin/env python3
"""Derive a safe NCCL_P2P_LEVEL from the host's GPU topology.

Prints exactly one NCCL_P2P_LEVEL token (NVL, PIX, PXB, PHB, or SYS) on
stdout; all diagnostics go to stderr. The derived level is the WORST pairwise
link class among the selected GPUs, so NCCL never attempts peer-to-peer over
a path the topology does not support (the dual-socket/PHB trap), and never
falls back to SYS on a workstation whose fabric supports PXB (the silent
prefill-throughput trap).

Launcher integration (derive only when unset or explicitly "auto"; an
explicit level always wins; derivation failure leaves NCCL to its own
defaults):

    if [ "${NCCL_P2P_LEVEL:-auto}" = "auto" ]; then
        if LEVEL=$(python3 /usr/local/bin/derive_nccl_p2p_level.py \
                       --devices "${CUDA_VISIBLE_DEVICES:-}"); then
            export NCCL_P2P_LEVEL="$LEVEL"
            echo "NCCL_P2P_LEVEL=auto -> derived ${LEVEL}" >&2
        else
            unset NCCL_P2P_LEVEL
            echo "NCCL_P2P_LEVEL derivation failed; leaving NCCL defaults" >&2
        fi
    fi

Exit codes: 0 = level printed; 2 = not applicable (fewer than two GPUs
selected); 3 = nvidia-smi unavailable or matrix unparseable.

`--self-test` runs against embedded fixture matrices (no GPU required).

Limits (deliberate): this is static derivation from `nvidia-smi topo -m`.
It cannot detect functionally broken P2P behind a healthy-looking link
class (ACS/IOMMU misconfiguration); that is the measured-probe extension
tracked for the #81 calibration framework.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# nvidia-smi link classes, best to worst, and the NCCL_P2P_LEVEL each maps
# to. NV# (NVLink, any lane count) maps to NVL. NODE (crossing host bridges
# within a NUMA node) has no NCCL token of its own; SYS is the safe cap.
_CLASS_RANK = {"NV": 0, "PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 5}
_CLASS_TO_LEVEL = {
    "NV": "NVL",
    "PIX": "PIX",
    "PXB": "PXB",
    "PHB": "PHB",
    "NODE": "SYS",
    "SYS": "SYS",
}


def parse_topo_matrix(text: str) -> dict[tuple[str, str], str]:
    """Return {(gpu_a, gpu_b): link_class} for every GPU pair in the matrix."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]

    def _is_link_token(t: str) -> bool:
        return t == "X" or t.startswith("NV") or t in _CLASS_RANK

    # The header is the first line that names GPU columns but contains no
    # link cells; data rows start with a GPU label followed by link cells.
    header = None
    header_idx = -1
    for i, ln in enumerate(lines):
        toks = ln.split()
        if (
            toks
            and any(re.fullmatch(r"GPU\d+", t) for t in toks)
            and not any(_is_link_token(t) for t in toks)
        ):
            header, header_idx = toks, i
            break
    if header is None:
        raise ValueError("no topology header row found")
    gpu_cols = [t for t in header if re.fullmatch(r"GPU\d+", t)]
    if not gpu_cols:
        raise ValueError("no GPU columns in topology header")

    pairs: dict[tuple[str, str], str] = {}
    for ln in lines[header_idx + 1 :]:
        toks = ln.split()
        if not toks or not re.fullmatch(r"GPU\d+", toks[0]):
            continue
        row = toks[0]
        cells = toks[1 : 1 + len(gpu_cols)]
        if len(cells) < len(gpu_cols):
            raise ValueError(f"short row for {row}")
        for col, cell in zip(gpu_cols, cells):
            if row == col:
                continue
            cls = "NV" if cell.startswith("NV") else cell
            if cls not in _CLASS_RANK and cell != "X":
                raise ValueError(f"unknown link class {cell!r} ({row}-{col})")
            if cell != "X":
                pairs[(row, col)] = cls
    return pairs


def derive_level(
    pairs: dict[tuple[str, str], str],
    devices: list[str],
    permissive: bool = False,
) -> tuple[str, tuple[str, str, str]]:
    """Derive the NCCL_P2P_LEVEL for the selected devices.

    NCCL_P2P_LEVEL is a permissiveness CAP: pairs at or below the level use
    direct P2P, pairs beyond it are routed through the CPU. The default
    policy derives the worst pairwise class but never exceeds PXB: P2P
    through PCIe switches is enabled, P2P through or above a PCIe Host
    Bridge (PHB/NODE/SYS pairs) is not -- those paths are where broken-P2P
    hangs and completion-timeout wedges live on commodity multi-GPU boxes.
    ``permissive=True`` lifts the cap to the literal worst class for hosts
    whose cross-bridge P2P is known-healthy."""
    worst = None
    worst_pair = None
    for i, a in enumerate(devices):
        for b in devices[i + 1 :]:
            cls = pairs.get((a, b)) or pairs.get((b, a))
            if cls is None:
                raise ValueError(f"no topology entry for pair {a}-{b}")
            if worst is None or _CLASS_RANK[cls] > _CLASS_RANK[worst]:
                worst, worst_pair = cls, (a, b, cls)
    assert worst is not None and worst_pair is not None
    level = _CLASS_TO_LEVEL[worst]
    if not permissive and _CLASS_RANK[worst] > _CLASS_RANK["PXB"]:
        level = "PXB"
    return level, worst_pair


def _selected_devices(arg: str | None, pairs) -> list[str]:
    all_gpus = sorted(
        {g for p in pairs for g in p}, key=lambda s: int(s[3:])
    )
    raw = (arg or "").strip()
    if not raw:
        return all_gpus
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(
                f"non-numeric device id {tok!r} (UUID selection unsupported)"
            )
        name = f"GPU{int(tok)}"
        if name not in all_gpus:
            raise ValueError(f"{name} not present in topology matrix")
        out.append(name)
    return out


_FIXTURES = {
    # 4x RTX PRO 6000 workstation, single root complex, paired switches
    # (the CN3/CN4 shape): expect PXB.
    "workstation-pxb": (
        """\
\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tPIX\tPXB\tPXB\t0-31\t0
GPU1\tPIX\t X \tPXB\tPXB\t0-31\t0
GPU2\tPXB\tPXB\t X \tPIX\t0-31\t0
GPU3\tPXB\tPXB\tPIX\t X \t0-31\t0
""",
        "PXB",
    ),
    # Dual-socket server, two GPUs per socket: expect SYS.
    "dual-socket-capped-pxb": (
        """\
\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tPXB\tSYS\tSYS\t0-23\t0
GPU1\tPXB\t X \tSYS\tSYS\t0-23\t0
GPU2\tSYS\tSYS\t X \tPXB\t24-47\t1
GPU3\tSYS\tSYS\tPXB\t X \t24-47\t1
""",
        "PXB",
    ),
    # NVLink mesh: expect NVL.
    "nvlink-nvl": (
        """\
\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNV4\tNV4\tNV4\t0-63\t0
GPU1\tNV4\t X \tNV4\tNV4\t0-63\t0
GPU2\tNV4\tNV4\t X \tNV4\t0-63\t0
GPU3\tNV4\tNV4\tNV4\t X \t0-63\t0
""",
        "NVL",
    ),
    # Mixed: NVLink pairs but PHB across pairs: worst wins -> PHB.
    "mixed-capped-pxb": (
        """\
\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNV2\tPHB\tPHB\t0-31\t0
GPU1\tNV2\t X \tPHB\tPHB\t0-31\t0
GPU2\tPHB\tPHB\t X \tNV2\t0-31\t0
GPU3\tPHB\tPHB\tNV2\t X \t0-31\t0
""",
        "PXB",
    ),
    # NODE caps to SYS (no NCCL NODE token).
    "node-capped-pxb": (
        """\
\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNODE\t0-31\t0
GPU1\tNODE\t X \t0-31\t0
""",
        "PXB",
    ),
}


def _self_test() -> int:
    failed = 0
    for name, (matrix, expect) in _FIXTURES.items():
        try:
            pairs = parse_topo_matrix(matrix)
            level, worst = derive_level(pairs, _selected_devices(None, pairs))
            ok = level == expect
        except Exception as exc:  # noqa: BLE001 - report and count
            print(f"FAIL {name}: exception {exc}", file=sys.stderr)
            failed += 1
            continue
        print(
            f"{'PASS' if ok else 'FAIL'} {name}: derived {level} "
            f"(expected {expect}; worst pair {worst[0]}-{worst[1]} {worst[2]})",
            file=sys.stderr,
        )
        failed += 0 if ok else 1
    # Permissive mode lifts the cap: dual-socket worst SYS -> SYS.
    pairs = parse_topo_matrix(_FIXTURES["dual-socket-capped-pxb"][0])
    level, _ = derive_level(pairs, _selected_devices(None, pairs), permissive=True)
    ok = level == "SYS"
    print(f"{'PASS' if ok else 'FAIL'} permissive-sys: derived {level} (expected SYS)",
          file=sys.stderr)
    failed += 0 if ok else 1
    # Subset selection: workstation matrix, switch-local pair only -> PIX.
    pairs = parse_topo_matrix(_FIXTURES["workstation-pxb"][0])
    level, _ = derive_level(pairs, _selected_devices("0,1", pairs))
    ok = level == "PIX"
    print(
        f"{'PASS' if ok else 'FAIL'} subset-0,1-pix: derived {level} "
        "(expected PIX)",
        file=sys.stderr,
    )
    failed += 0 if ok else 1
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        help="comma-separated GPU indices (default: CUDA_VISIBLE_DEVICES, "
        "else all GPUs)",
    )
    ap.add_argument(
        "--permissive",
        action="store_true",
        help="lift the PXB cap: derive the literal worst link class",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    # An explicitly configured level ALWAYS wins: echo it, derive nothing.
    explicit = os.environ.get("NCCL_P2P_LEVEL", "").strip()
    if explicit and explicit.lower() != "auto":
        print(
            f"derive_nccl_p2p_level: NCCL_P2P_LEVEL={explicit} set "
            "explicitly; respecting it", file=sys.stderr,
        )
        print(explicit)
        return 0

    try:
        text = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - single fail-safe boundary
        print(f"derive_nccl_p2p_level: nvidia-smi failed: {exc}", file=sys.stderr)
        return 3

    try:
        pairs = parse_topo_matrix(text)
        devices = _selected_devices(args.devices, pairs)
        if len(devices) < 2:
            print(
                "derive_nccl_p2p_level: fewer than two GPUs selected; "
                "P2P level not applicable",
                file=sys.stderr,
            )
            return 2
        level, worst = derive_level(pairs, devices, permissive=args.permissive)
    except ValueError as exc:
        print(f"derive_nccl_p2p_level: {exc}", file=sys.stderr)
        return 3

    print(
        f"derive_nccl_p2p_level: devices={','.join(devices)} "
        f"worst-pair={worst[0]}-{worst[1]}({worst[2]}) -> {level}",
        file=sys.stderr,
    )
    print(level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
