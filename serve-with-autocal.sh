#!/bin/bash
# Auto-calibration entrypoint wrapper for GLM-5.2 v20 serving.
#
# Runs the NCCL_P2P_LEVEL auto hook BEFORE the serving stack initializes
# NCCL, then execs the stock serve script. Contract:
#   - explicit NCCL_P2P_LEVEL (anything but "auto"/empty): respected
#     verbatim, no measurement (also enforced inside the tools themselves);
#   - "auto" or unset: measured probe first (pre-model-load, GPUs empty),
#     static topology derivation on probe failure, stack defaults last;
#   - every decision logged to stderr; probe results cached under
#     /root/.cache/nccl-p2p-probe (fingerprint-keyed).
#
# Wire-protocol auto tokens (F8_DMA=auto / i8_auto / mx_auto) are documented
# in design/measured-comm-calibration-spec-20260728.md and land with the
# SparkInfer #81 probe extension; this wrapper does not alter F8_DMA.
set -u

_autocal_log() { echo "[autocal] $*" >&2; }

case "${NCCL_P2P_LEVEL:-auto}" in
  auto|"")
    LEVEL=""
    if LEVEL=$(python3 /usr/local/bin/nccl_p2p_probe.py \
                 --devices "${CUDA_VISIBLE_DEVICES:-}" 2> >(sed 's/^/[autocal] /' >&2)); then
      _autocal_log "NCCL_P2P_LEVEL=auto -> measured ${LEVEL}"
    elif LEVEL=$(python3 /usr/local/bin/derive_nccl_p2p_level.py \
                 --devices "${CUDA_VISIBLE_DEVICES:-}" 2> >(sed 's/^/[autocal] /' >&2)); then
      _autocal_log "NCCL_P2P_LEVEL=auto -> probe unavailable, static ${LEVEL}"
    else
      _autocal_log "NCCL_P2P_LEVEL=auto -> derivation unavailable, leaving NCCL defaults"
      LEVEL=""
    fi
    if [ -n "${LEVEL}" ]; then
      export NCCL_P2P_LEVEL="${LEVEL}"
    else
      unset NCCL_P2P_LEVEL
    fi
    ;;
  *)
    _autocal_log "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL} set explicitly; respected"
    ;;
esac

exec /usr/local/bin/serve-gilded-gnosis.sh "$@"
