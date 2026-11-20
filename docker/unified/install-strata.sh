#!/bin/bash
# Install Strata (source layout engine/ + terminal/) - clone, build the engine
# (sm_86, Tesla A2 pair) and its Python venv.
#
# Since 0.2.16 the Python side (setup.py, requirements*.txt, serve/, strata_cli/)
# lives in terminal/ and ggml is vendored in engine/third_party/strata-ggml, so
# nothing is fetched from llama.cpp any more. The build follows Strata's own
# tools/docker/Dockerfile.src (same cmake configure line), so a later setup.py
# run inside the runtime container recognises the engine and skips recompiling.
#
# Usage: ./install-strata.sh <commit_hash_or_tag>
# Env:   CMAKE_CUDA_ARCHITECTURES (from the image, default 86 = sm_86/A2/3060/3090)
#        STRATA_BUILD_VISION=1 also builds the CPU image encoder (default 0: text
#        only - the first-setup here runs `--vision no`)
#        STRATA_BUILD_SPLIT=1 also builds strata-split (2-4 cards; it never
#        activates on a 2x A2 pair, so off by default: it only costs build time)
set -e

STRATA_REF="${1:-v0.2.17}"
STRATA_REPO="https://github.com/Niko1221/Strata.git"
STRATA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-86}"
STRATA_VISION="${STRATA_BUILD_VISION:-0}"
STRATA_SPLIT="${STRATA_BUILD_SPLIT:-0}"
SRC=/src/Strata
VENV="${SRC}/terminal/.venv"

# exported: the staging heredoc below runs as a child process and reads these
export STRATA_ARCH STRATA_VISION STRATA_SPLIT

echo "=== Strata ${STRATA_REF} (arch ${STRATA_ARCH}, vision ${STRATA_VISION}, split ${STRATA_SPLIT}) ==="

# --- source (init-based fetch so a cached /src/Strata dir doesn't break clone) ---
mkdir -p "$SRC"
cd "$SRC"
if [ ! -d .git ]; then
    git init
    git remote add origin "${STRATA_REPO}"
fi
git fetch --depth=1 origin "${STRATA_REF}"
git checkout -f FETCH_HEAD
# The image needs the sources, not their history: a shallow .git is still a few
# hundred MB in the layer, and the staging copy below leaves it out.
rm -rf "$SRC/.git"

# --- the server's venv + the build tools (cmake/ninja come from
#     requirements-build.txt: a pip build needs no system cmake) ---
python3 -m venv "$VENV"
"$VENV/bin/pip" install --no-cache-dir --upgrade pip
"$VENV/bin/pip" install --no-cache-dir \
    -r "${SRC}/terminal/requirements.txt" \
    -r "${SRC}/terminal/requirements-build.txt"

# --- engine: the same configure line as Strata's tools/docker/Dockerfile.src.
#     "86-real;86-virtual": machine code for sm_86 plus PTX for later cards.
#     STRATA_PORTABLE=ON keeps the AVX2 floor: the runner CPU has AVX-512, the
#     Unraid box's Intel ES sample only AVX2 - a native build dies with SIGILL. ---
cd "$SRC"
real=""
for a in $(echo "${STRATA_ARCH}" | tr ';,' '  '); do real="${real};${a}-real"; done
real="${real#;}"
last=$(echo "${STRATA_ARCH}" | tr ';,' '  ' | awk '{print $NF}')
sm60=OFF
for a in $(echo "${STRATA_ARCH}" | tr ';,' '  '); do
    if [ "$a" -lt 75 ]; then sm60=ON; fi
done

"$VENV/bin/cmake" -G Ninja -S engine -B build -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_EXPERIMENTAL_SM60=$sm60 \
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"
# The build tree, not the configure line, decides whether strata_kq exists.
. tools/packaging/engine_targets.sh
targets=$(engine_targets "$VENV/bin/cmake" build)
"$VENV/bin/cmake" --build build --target $targets -j "$(nproc)"

if [ "$STRATA_VISION" = "1" ]; then
    "$VENV/bin/cmake" -G Ninja -S terminal/tools/vision -B build-vision \
        -DCMAKE_BUILD_TYPE=Release -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON
    "$VENV/bin/cmake" --build build-vision --target strata-vision -j "$(nproc)"
fi

if [ "$STRATA_SPLIT" = "1" ]; then
    "$VENV/bin/cmake" -G Ninja -S engine -B build-split \
        -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
        -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_SPLIT=ON -DSTRATA_PORTABLE=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"
    "$VENV/bin/cmake" --build build-split --target strata-split -j "$(nproc)"
fi

# --- stage the engine beside the Terminal, in the folder engine_dir() names
#     (ROOT/engine = terminal/engine), and write the BUILD.json the runtime
#     checks: source "local" + the source hash == no recompile at first start.
#     The constants and both hashes are reproduced from setup.py itself (read
#     with ast, not imported: its top level installs TLS and pulls in
#     strata_control, which has no place in a build stage) - a hand-written
#     value that drifts would make the first container start try to recompile. ---
cd "$SRC/terminal"
"$VENV/bin/python" - <<'PYEOF'
import ast, hashlib, json, os, pathlib, re, shutil

src = pathlib.Path("/src/Strata")
terminal = src / "terminal"
engine = src / "engine"
archs = [int(a.split("-")[0]) for a in os.environ["STRATA_ARCH"].split(";") if a.split("-")[0].isdigit()]

tree = ast.parse((terminal / "setup.py").read_text(encoding="utf-8"))
consts = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ("LLAMA_CPP_COMMIT", "ENGINE_SOURCES", "VISION_SOURCES"):
            consts[node.targets[0].id] = ast.literal_eval(node.value)
for need in ("LLAMA_CPP_COMMIT", "ENGINE_SOURCES", "VISION_SOURCES"):
    if need not in consts:
        raise SystemExit(f"setup.py no longer defines {need}: update install-strata.sh")

def source_hash(parts):
    """setup.py's source_hash(): the files an engine is built from, in order."""
    h = hashlib.sha256(consts["LLAMA_CPP_COMMIT"].encode())
    for part in parts:
        top = terminal if part.startswith("tools/") else engine
        base = top / part
        files = [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file())
        for f in files:
            h.update(f.relative_to(top).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]

def source_version():
    """setup.py's source_version(): the project version engine/CMakeLists.txt builds."""
    try:
        m = re.search(r"project\(strata VERSION ([\d.]+)", (engine / "CMakeLists.txt").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return "0"
    return m.group(1) if m else "0"

eng = terminal / "engine"
eng.mkdir(parents=True, exist_ok=True)
shutil.copy2(src / "build" / "strata", eng / "strata")
kq = src / "build" / "libstrata_kq.so"
if kq.exists():
    shutil.copy2(kq, eng / "libstrata_kq.so")

vision = "none"
if os.environ.get("STRATA_VISION") == "1":
    shutil.copy2(src / "build-vision" / "bin" / "strata-vision", eng / "strata-vision")
    vision = "cpu"

split_note = ""
if os.environ.get("STRATA_SPLIT") == "1":
    split_bin = src / "build-split" / "src" / "split" / "strata-split"
    if split_bin.exists():
        shutil.copy2(split_bin, eng / "strata-split")
    else:
        split_note = " (strata-split binary not found, skipped)"

version = source_version()
if version == "0":
    raise SystemExit("engine/CMakeLists.txt carries no project(strata VERSION x.y.z): cannot stamp BUILD.json")

meta = {
    "source": "local",
    "version": version,
    "archs": archs,
    "vision": vision,
    "src": source_hash(consts["ENGINE_SOURCES"]),
    "vision_src": source_hash(consts["VISION_SOURCES"]) if vision != "none" else None,
}
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
print(f"engine staged: {eng / 'strata'} (version {version}, archs {archs}, vision {vision}){split_note}")
PYEOF

# --- stage the install tree (build dirs are build-time only) ---
cd "$SRC"
rm -rf build build-vision build-split
mkdir -p /install/strata
cp -a "$SRC"/. /install/strata/

# The venv was created at /src/Strata during the build, so its console scripts
# (cmake, ninja, ...) carry that path in their shebang and fail with "not found"
# once the tree lives at /opt/strata. Rewrite them (text scripts only - the loop
# skips the ELF binaries and the python symlinks).
for f in /install/strata/terminal/.venv/bin/*; do
    if head -1 "$f" 2>/dev/null | grep -q "^#!/src/Strata/terminal/.venv/bin/python"; then
        sed -i "1s|.*|#!/opt/strata/terminal/.venv/bin/python3|" "$f"
    fi
done
echo "=== Strata build complete ==="
ls -la /install/strata/terminal/engine/
