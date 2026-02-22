#!/usr/bin/env bash
set -euo pipefail

# ---------- helpers ----------
first_ipv4() {
  # hostname -i can return multiple; pick the first IPv4-looking token.
  hostname -i | tr ' ' '\n' | awk '/^([0-9]{1,3}\.){3}[0-9]{1,3}$/{print; exit}'
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------- cluster identity ----------
NODE_RANK="${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}"

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-${MASTER_ADDR:-}}"
if [[ -z "${RAY_HEAD_ADDR}" ]]; then
  RAY_HEAD_ADDR="127.0.0.1"
fi

THIS_IP="${THIS_IP:-$(first_ipv4 || true)}"
if [[ -z "${THIS_IP}" ]]; then
  THIS_IP="127.0.0.1"
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Enable our runtime patches (applies in Ray workers too via sitecustomize.py)
export ASYNC_RL_ENABLE_PATCHES="${ASYNC_RL_ENABLE_PATCHES:-1}"
export ASYNC_RL_ENABLE_MILES_FSDP_PATCH="${ASYNC_RL_ENABLE_MILES_FSDP_PATCH:-1}"
export ASYNC_RL_ENABLE_HF_KERNELS="${ASYNC_RL_ENABLE_HF_KERNELS:-1}"
export ASYNC_RL_PATCH_FA3_SINK_BWD="${ASYNC_RL_PATCH_FA3_SINK_BWD:-1}"
export ASYNC_RL_KERNEL_DEVICE="${ASYNC_RL_KERNEL_DEVICE:-cuda}"

cleanup() {
  # Best-effort Ray shutdown
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# ---------- start / join ray ----------
if [[ -n "${RAY_ADDRESS:-}" ]]; then
  echo "[entrypoint] Using existing Ray cluster: RAY_ADDRESS=${RAY_ADDRESS}"
else
  if [[ "${NODE_RANK}" == "0" ]]; then
    echo "[entrypoint] Starting Ray head on ${RAY_HEAD_ADDR}:${RAY_PORT} (this_ip=${THIS_IP})"
    ray start \
      --head \
      --node-ip-address "${THIS_IP}" \
      --port "${RAY_PORT}" \
      --dashboard-host "0.0.0.0" \
      --dashboard-port "${RAY_DASHBOARD_PORT}" \
      --num-gpus "${NUM_GPUS_PER_NODE}" \
      --disable-usage-stats
  else
    echo "[entrypoint] Joining Ray cluster at ${RAY_HEAD_ADDR}:${RAY_PORT} (this_ip=${THIS_IP})"
    ray start \
      --address "${RAY_HEAD_ADDR}:${RAY_PORT}" \
      --node-ip-address "${THIS_IP}" \
      --num-gpus "${NUM_GPUS_PER_NODE}" \
      --disable-usage-stats
  fi
  export RAY_ADDRESS="${RAY_HEAD_ADDR}:${RAY_PORT}"
fi

# ---------- run driver on head ----------
if [[ "${NODE_RANK}" == "0" ]]; then
  echo "[entrypoint] Launching Miles driver (head node)."
  exec python -m async_rl.miles_train "$@"
else
  echo "[entrypoint] Ray worker node running; blocking."
  exec tail -f /dev/null
fi

