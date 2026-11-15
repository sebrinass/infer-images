#!/bin/bash
# Install FastLLM (ztxz16/fastllm) into its own isolated venv.
#
# Why isolated: ftllm ships ONE prebuilt Linux wheel compiled against CUDA
# 12.9, and its metadata pulls nvidia-*-cu12 runtime wheels (cudart / cublas /
# nccl). This image's runtime is CUDA 13.3; vLLM, SGLang and Strata each carry
# their own CUDA stack too. Keeping ftllm in /opt/fastllm-venv and NEVER adding
# it to the global PATH or LD_LIBRARY_PATH means its CUDA 12 libraries cannot
# shadow or disturb any other backend.
#
# Version: build-image.sh resolves the latest ftllm release from PyPI at build
# time and passes it in as $1. Nothing is pinned in this tree and no
# constraints file is used, so new upstream releases flow in on the next
# build. Injecting the resolved version as a build-arg also busts the BuildKit
# layer cache when upstream publishes a new release.
#
# Usage: ./install-fastllm.sh <version|latest>
set -euo pipefail

FTLLM_VERSION="${1:-latest}"
VENV="/opt/fastllm-venv"
SITE="${VENV}/lib/python3.12/site-packages"

echo "=== FastLLM (ftllm ${FTLLM_VERSION}) -> ${VENV} ==="

uv venv "${VENV}" --python 3.12 --seed
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:${PATH}"
export UV_NO_CACHE=1

if [[ "${FTLLM_VERSION}" == "latest" || -z "${FTLLM_VERSION}" ]]; then
    uv pip install --upgrade ftllm
else
    uv pip install --upgrade "ftllm==${FTLLM_VERSION}"
fi

# Launcher. ftllm's native .so files need the CUDA 12 libraries that its own
# pip dependencies installed under site-packages/nvidia/*/lib. Prepend those
# paths for THIS PROCESS ONLY (exec'd child), so the image-wide
# LD_LIBRARY_PATH stays exactly as it was for llama.cpp / vLLM / SGLang /
# Strata. The glob also tolerates future pip package layout changes.
cat > "${VENV}/bin/ftllm-launch" <<'LAUNCHER'
#!/bin/bash
set -e
VENV=/opt/fastllm-venv
SITE="$VENV/lib/python3.12/site-packages"
LIBS=""
if [ -d "$SITE/nvidia" ]; then
    for d in "$SITE"/nvidia/*/lib; do
        [ -d "$d" ] || continue
        LIBS="${LIBS:+$LIBS:}$d"
    done
fi
export LD_LIBRARY_PATH="${LIBS:+${LIBS}:}${LD_LIBRARY_PATH:-}"
exec "$VENV/bin/ftllm" "$@"
LAUNCHER
chmod +x "${VENV}/bin/ftllm-launch"

# Trim anything uv left behind; the wheel itself stays in the venv.
rm -rf /root/.cache/uv /root/.cache/pip

# Metadata-only check (importing ftllm here would try to initialise CUDA and
# there is no GPU on the build runner).
"${VENV}/bin/python" -c \
    "import importlib.metadata as m; print('ftllm installed:', m.version('ftllm'))"
test -x "${VENV}/bin/ftllm"
test -x "${VENV}/bin/ftllm-launch"
echo "=== FastLLM install complete ==="
