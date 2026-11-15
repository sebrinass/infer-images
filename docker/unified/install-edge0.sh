#!/bin/bash
# Install Edge0 - the edge0-strata engine (a fork of Strata with latent-MoE
# support) plus its Python HTTP server, for the Edge0 145B / 120B models.
#
# Why a separate component and not a newer Strata: edge0-strata is a fork whose
# latent-MoE kernels (NVFP4 / Q3_K experts at latent width, per-layer D/U
# projections) upstream Strata does not have. The two are installed side by
# side and never share a path, so either can be disabled without touching the
# other:
#
#   /opt/strata        upstream Strata   (Qwen3.8-Flash-Next packs)
#   /opt/edge0         edge0-strata      (Edge0 145B / 120B latent-MoE packs)
#
# The model packs themselves are NOT in the image (40-100 GB each); they are
# expected at a mount such as /models/Edge0-120B-q3k. See /opt/edge0/run.
#
# Usage: ./install-edge0.sh <commit_hash_or_tag>   (default: main)
# Env:   CMAKE_CUDA_ARCHITECTURES (from the image, default 86 = sm_86/A2)
#        EDGE0_BUILD_VISION=0 skips tools/vision when the fork ships it
#        (default 1: build it if present and let a failure only warn - text
#        inference does not depend on it)
set -e

EDGE0_REF="${1:-main}"
EDGE0_REPO="https://github.com/Edge0-AI/Edge0.git"
# llama.cpp commit the fork's engine builds against (same pin EDGE0.md and
# install-strata.sh use); the engine takes ggml from it via -DSTRATA_GGML_DIR.
LLAMA_CPP_COMMIT="3cf03257f219afbe7334045ff7c6a06ac68c627d"

EDGE0_ARCH="${CMAKE_CUDA_ARCHITECTURES:-86}"
EDGE0_VISION="${EDGE0_BUILD_VISION:-1}"
# exported: the build below reads them in subshells
export NVCC="$(command -v nvcc)"
export EDGE0_ARCH EDGE0_VISION

EDGE0_SRC="/src/Edge0"
EDGE0_OUT="/opt/edge0"
STRATA_SRC="${EDGE0_SRC}/edge0-strata"

echo "=== Edge0 ${EDGE0_REF} (arch ${EDGE0_ARCH}, vision ${EDGE0_VISION}) ==="

# --- source (init-based fetch so a cached /src/Edge0 does not break the clone) ---
mkdir -p "${EDGE0_SRC}"
cd "${EDGE0_SRC}"
if [ ! -d .git ]; then
    git init
    git remote add origin "${EDGE0_REPO}"
fi
git fetch --depth=1 origin "${EDGE0_REF}"
git checkout -f FETCH_HEAD

# --- ggml sources at the pinned llama.cpp commit ---
mkdir -p /src/llama.cpp-edge0
cd /src/llama.cpp-edge0
if [ ! -d .git ]; then
    git init
    git remote add origin https://github.com/ggml-org/llama.cpp.git
fi
git fetch --depth=1 origin "${LLAMA_CPP_COMMIT}"
git checkout -f FETCH_HEAD

# --- the engine, exactly the flags EDGE0.md documents for a Linux NVIDIA build ---
# PORTABLE=ON keeps the AVX2 floor: the cloud runner has AVX-512, the Unraid box
# does not, and a native build there dies with SIGILL (same reason
# install-strata.sh sets it).
cmake -S "${STRATA_SRC}" -B "${STRATA_SRC}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_CUDA_ARCHITECTURES="${EDGE0_ARCH}" \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_GGML_DIR=/src/llama.cpp-edge0 \
    -DSTRATA_PORTABLE=ON
cmake --build "${STRATA_SRC}/build" --target strata -j"$(nproc)"

# --- optional image encoder (tools/vision). Edge0 145B ships an mmproj file;
# text inference does not need it, so a failure here only warns. ---
if [ "${EDGE0_VISION}" = "1" ] && [ -d "${STRATA_SRC}/tools/vision" ]; then
    if ! cmake -S "${STRATA_SRC}/tools/vision" -B "${STRATA_SRC}/build-vision" -G Ninja \
            -DLLAMA_DIR=/src/llama.cpp-edge0 \
            -DSTRATA_VISION_CUDA=OFF \
            -DSTRATA_PORTABLE=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_COMPILER="${NVCC}" \
        || ! cmake --build "${STRATA_SRC}/build-vision" -j"$(nproc)"; then
        echo "WARN: strata-vision build failed - images stay off (text inference unaffected)" >&2
    fi
fi

# --- stage the tree at the path the runtime uses. The venv is created in place
# so its console-script shebangs point at /opt/edge0 and survive the COPY. ---
mkdir -p "${EDGE0_OUT}"
rm -rf "${EDGE0_OUT}/edge0-strata"
git -C "${EDGE0_SRC}" archive HEAD edge0-strata | tar -x -C "${EDGE0_OUT}"

# keep only the built binary in build/ - the object files are build-time only
mkdir -p "${EDGE0_OUT}/edge0-strata/build"
cp -a "${STRATA_SRC}/build/strata" "${EDGE0_OUT}/edge0-strata/build/strata"
for f in "${STRATA_SRC}"/build-vision/bin/strata-vision "${STRATA_SRC}"/build-vision/strata-vision; do
    if [ -f "${f}" ]; then
        cp -a "${f}" "${EDGE0_OUT}/edge0-strata/build/strata-vision"
        break
    fi
done

# --- the server's Python environment (requirements.txt of the fork) ---
python3 -m venv "${EDGE0_OUT}/edge0-strata/.venv"
"${EDGE0_OUT}/edge0-strata/.venv/bin/pip" install --no-cache-dir --upgrade pip
"${EDGE0_OUT}/edge0-strata/.venv/bin/pip" install --no-cache-dir \
    -r "${EDGE0_OUT}/edge0-strata/requirements.txt"

echo "=== Edge0 build complete ==="
ls -la "${EDGE0_OUT}/edge0-strata/build/"
"${EDGE0_OUT}/edge0-strata/.venv/bin/python" -c \
    "import numpy, jinja2, yaml, requests, tqdm, PIL, psutil; print('edge0 venv ok')"
