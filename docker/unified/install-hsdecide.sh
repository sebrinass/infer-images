#!/usr/bin/env bash
# hs-decide-1-flash: Hindsight's 0.8B decision model.
#
# Unlike every other model in this image, this one is not a generator. Hindsight
# hands it a state plus typed questions (choice / score / noul) and reads back a
# probability per option from a single forward pass. That shape is why neither
# vLLM nor llama.cpp is involved: there is no decoding loop to schedule, only
# `model.to(device)` and one pass over the classifier columns.
#
# This stage only FETCHES the vendor's serving package. Installing it is the
# runtime stage's job (see the Dockerfile), because the package needs a torch to
# run on and the only torch in this image is the CUDA build inside /opt/vllm-venv.
# That reuse is deliberate: a 0.8B single-forward model on the GPU answers in
# ~0.39 s with its candidates attached, against 31 s on the CPU cores, and it
# costs the image nothing - the wheel is ~50 KB.
#
# The 1.5 GB safetensors stay out of the image, mounted at runtime under /models
# like every other weight file.
set -euo pipefail

SRC=/opt/hsdecide-src
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
# Throwaway venv: it exists only to hold `hf` for the download, and is removed
# before this stage is copied anywhere.
FETCH=/tmp/hsdecide-fetch

python3.12 -m venv "${FETCH}"
"${FETCH}/bin/pip" install --upgrade pip >/dev/null
"${FETCH}/bin/pip" install "huggingface_hub[cli]>=0.34"

echo "hs-decide: fetching the serving package from ${HF_ENDPOINT}"
HF_ENDPOINT="${HF_ENDPOINT}" "${FETCH}/bin/hf" download \
    hindsight-ai/hs-decide-1-flash \
    --include "serving/*" --local-dir "${SRC}"

rm -rf "${FETCH}"

# The launcher the llama-swap entry calls. Kept as a thin wrapper so the config
# never has to know which venv carries the package, and so verify can test one
# stable name.
cat > /usr/local/bin/hs-decide-serve <<'SH'
#!/bin/sh
exec /opt/vllm-venv/bin/hindsight-decision-serve-transformers "$@"
SH
chmod 755 /usr/local/bin/hs-decide-serve

echo "hs-decide: serving package vendored at ${SRC}/serving"
