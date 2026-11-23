#!/usr/bin/env bash
# hs-decide-1-flash: Hindsight's 0.8B decision model.
#
# Unlike every other model in this image, this one is not a generator. Hindsight
# hands it a state plus typed questions (choice / score / noul) and reads back a
# probability per option from a single forward pass. That shape is why neither
# vLLM nor llama.cpp is involved: there is no decoding loop to schedule, only
# `model.to(device)` and one pass over the classifier columns.
#
# What goes into the image: the vendor's `serving/` package (a few hundred KB,
# adds POST /v1/systemone). What stays out: the 1.5 GB safetensors, mounted at
# runtime under /models like every other weight file.
#
# Consumers: Hindsight's consolidation triage (the only path that is worth
# wiring today), and optionally reflect / pages - both of which are on the
# interactive path, so judge them on latency before enabling.
set -euo pipefail

VENV=/opt/hsdecide-venv
SRC=/opt/hsdecide-src
# CPU build only. The triage path this serves is a background consolidation
# job: ~2.2 s per decision on CPU against ~0.09 s on an Arc iGPU, and nobody is
# waiting on it. CPU torch is ~200 MB against ~2.5 GB for the cu126 wheel, and
# it keeps this venv out of the GPU memory budget entirely.
# To move to a GPU later, change this to .../whl/cu126 (or cu12x) and set
# CUDA_VISIBLE_DEVICES in the llama-swap entry - nothing else changes.
TORCH_INDEX="${HSDECIDE_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

python3.12 -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip wheel >/dev/null

echo "hs-decide: installing torch from ${TORCH_INDEX}"
"${VENV}/bin/pip" install --index-url "${TORCH_INDEX}" torch

"${VENV}/bin/pip" install "huggingface_hub[cli]>=0.34"

echo "hs-decide: fetching the serving package from ${HF_ENDPOINT}"
HF_ENDPOINT="${HF_ENDPOINT}" "${VENV}/bin/hf" download \
    hindsight-ai/hs-decide-1-flash \
    --include "serving/*" --local-dir "${SRC}"

"${VENV}/bin/pip" install "${SRC}/serving[server]"

# The launcher the llama-swap entry calls. Kept as a thin wrapper so the config
# never has to know the venv layout, and so verify can test one stable name.
cat > /usr/local/bin/hs-decide-serve <<'SH'
#!/bin/sh
exec /opt/hsdecide-venv/bin/hindsight-decision-serve-transformers "$@"
SH
chmod 755 /usr/local/bin/hs-decide-serve

# Import check: proves torch and the vendor package resolve. The weights are
# deliberately absent, so nothing here touches the model itself.
"${VENV}/bin/python" - <<'PY'
import torch, transformers, hindsight_decision
print(
    "hs-decide venv ok (cpu-only build):",
    f"torch {torch.__version__}",
    f"transformers {transformers.__version__}",
    f"cuda compiled-in: {torch.version.cuda or 'no'}",
)
PY
