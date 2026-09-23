#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

# Submit through scripts/entrypoint/ray-job.sh against a local Ray cluster.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/../../models/qwen3-0.6B.sh"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/model}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
SMOKE_MODE="${SMOKE_MODE:-grpo}"
RUN_NAME="${RUN_NAME:-qwen3-0.6b-${SMOKE_MODE}-smoke-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${PROJECT_ROOT}/log"

RESOURCE='{"actor": [1, 1], "rollout": [1, 1]}'
SCORING_ARGS=()
case "${SMOKE_MODE}" in
    grpo) ;;
    defer-genrm)
        RESOURCE='{"actor": [1, 1], "rollout": [1, 1], "genrm": [1, 1]}'
        SCORING_ARGS=(
            --rm-type dummy --defer-reward-to-post-process
            --custom-reward-post-process-path examples.generate_reward_model.post_process_genrm_swap.custom_reward_post_process
            --genrm-model-path "${MODEL_DIR}/Qwen3-0.6B"
            --genrm-num-gpus 1 --genrm-num-gpus-per-engine 1
            --genrm-engine-config '{"context_length":2048,"mem_fraction_static":0.5}'
            --genrm-sampling-config '{"temperature":0,"max_response_len":32,"chat_template_kwargs":{"enable_thinking":false}}'
        )
        ;;
    defer-opd|defer-opd-union)
        RESOURCE='{"actor": [1, 1], "rollout": [1, 1], "teacher": [1, 1]}'
        SCORING_ARGS=(
            --use-opd --opd-type sglang
            --teacher-hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B"
            --teacher-num-gpus-per-engine 1
            --teacher-sglang-mem-fraction-static 0.5
            --teacher-sglang-context-length 2048
            --opd-kl-coef 0.1 --opd-loss-coef 0
        )
        if [[ "${SMOKE_MODE}" == defer-opd-union ]]; then
            SCORING_ARGS+=(--opd-token-selection union --opd-log-prob-top-k 4)
        fi
        ;;
    *) echo "Unknown SMOKE_MODE: ${SMOKE_MODE}" >&2; exit 1 ;;
esac

# All service traffic in this local smoke run must bypass inherited proxies.
SMOKE_RUNTIME_ENV="$(python - "${SMOKE_MODE}" <<'PY'
import json
import os
import sys

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime.setdefault("env_vars", {}).update({"NO_PROXY": "*", "no_proxy": "*"})
if sys.argv[1] == "defer-opd-union":
    runtime["env_vars"].update({"RELAX_OPD_PER_POS_TOKEN_IDS": "1", "RELAX_OPD_TOKEN_IDS_LOGPROB_K": "4"})
print(json.dumps(runtime))
PY
)"

ray job submit --no-wait --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    --submission-id "${RUN_NAME}" \
    --runtime-env-json="${SMOKE_RUNTIME_ENV}" \
    -- python3 -m relax.entrypoints.train \
    --resource "${RESOURCE}" --rollout-num-gpus 1 \
    --num-gpus-per-node 1 --actor-num-gpus-per-node 1 \
    --no-enable-affinity --colocate --offload \
    --num-data-storage-units 1 --max-staleness 0 \
    --hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B" \
    --megatron-to-hf-mode bridge \
    --prompt-data "${DATA_DIR}/gsm8k/train.jsonl" \
    --input-key question --label-key answer --apply-chat-template \
    --apply-chat-template-kwargs '{"enable_thinking":false}' \
    --system-prompt 'Solve the problem with concise reasoning. End with a separate line exactly in the format Answer: <number>. Do not use units, markdown, or boxed notation on that final line.' \
    --rm-type dapo --reward-key score \
    --save-debug-rollout-data "${PROJECT_ROOT}/log/${RUN_NAME}-rollout-{rollout_id}.pt" \
    --num-rollout "${NUM_ROLLOUT:-3}" \
    --rollout-batch-size 8 --n-samples-per-prompt 8 --global-batch-size 64 \
    --rollout-max-response-len 1024 --rollout-max-prompt-len 512 \
    --rollout-max-context-len 2048 --rollout-temperature 1 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu 2048 \
    --recompute-granularity selective \
    --advantage-estimator grpo --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0 \
    --eps-clip 0.2 --eps-clip-high 0.28 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant \
    --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.98 \
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.5 \
    --attention-dropout 0 --hidden-dropout 0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --tb-project-name Relax/dev/inference-smoke --tb-experiment-name "${RUN_NAME}" \
    "${MODEL_ARGS[@]}" "${SCORING_ARGS[@]}" "$@" 2>&1 | tee "${PROJECT_ROOT}/log/${RUN_NAME}-submit.log"
