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
#
# Inkling-2-Mini's strata-inkling target is built with the main engine and staged
# as its own component under components/shapes/inkling2/engine/.
set -e

STRATA_REF="${1:-latest}"
STRATA_REPO="https://github.com/Niko1221/Strata.git"
STRATA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-86}"
STRATA_VISION="${STRATA_BUILD_VISION:-0}"
STRATA_SPLIT="${STRATA_BUILD_SPLIT:-0}"
SRC=/src/Strata
VENV="${SRC}/terminal/.venv"

# exported: the staging heredoc below runs as a child process and reads these
export STRATA_ARCH STRATA_VISION STRATA_SPLIT

# Never pin Strata: an empty or "latest" ref is resolved to the newest release
# tag here, so even a direct call (or a build that forgets to pass a hash) picks
# up the current engine instead of a stale hardcoded one.
if [ -z "${STRATA_REF}" ] || [ "${STRATA_REF}" = "latest" ] || [ "${STRATA_REF}" = "release" ]; then
    STRATA_REF=$(curl -fsSL https://api.github.com/repos/Niko1221/Strata/releases/latest \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])' 2>/dev/null) \
        || { echo "WARN: could not resolve latest Strata release, falling back to main HEAD" >&2; STRATA_REF=main; }
    echo "strata: resolved latest release -> ${STRATA_REF}"
fi

echo "=== Strata ${STRATA_REF} (arch ${STRATA_ARCH}, vision ${STRATA_VISION}, split ${STRATA_SPLIT}, inkling 1) ==="

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

# cmake is called by absolute path below, but it looks for its generator (`-G
# Ninja`) on PATH - and pip's ninja lands in $VENV/bin, not in /usr/bin. Without
# this, configure dies at once with "unable to find a build program
# corresponding to Ninja" and then a misleading "CMAKE_CXX_COMPILER not set".
# Strata's own tools/docker/Dockerfile.src carries the same line (PATH=/venv/bin).
export PATH="$VENV/bin:$PATH"
NINJA_BIN="$(command -v ninja || true)"
[ -n "$NINJA_BIN" ] || { echo "ERROR: ninja missing from the venv - requirements-build.txt changed?"; exit 1; }
echo "--- ninja: ${NINJA_BIN} ($("$NINJA_BIN" --version 2>/dev/null || echo '?'))"

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

#     -DGGML_CCACHE=OFF: builder-base puts /usr/lib/ccache first on PATH, and
#     the vendored ggml defaults GGML_CCACHE=ON, which wraps every compile in a
#     global RULE_LAUNCH_COMPILE - nvcc's fatbinary step then fails ("Could not
#     open input file 'device.ptx'"). setup.py turns it off for its own builds;
#     this one has to say it too.
"$VENV/bin/cmake" -G Ninja -S engine -B build -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_MAKE_PROGRAM=${NINJA_BIN}" \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_EXPERIMENTAL_SM60=$sm60 -DSTRATA_BUILD_INKLING=ON \
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON \
    -DGGML_CCACHE=OFF \
    "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"
# The build tree, not the configure line, decides whether strata_kq exists.
. tools/packaging/engine_targets.sh
targets=$(engine_targets "$VENV/bin/cmake" build)
"$VENV/bin/cmake" --build build --target $targets strata-inkling -j "$(nproc)"

if [ "$STRATA_VISION" = "1" ]; then
    "$VENV/bin/cmake" -G Ninja -S terminal/tools/vision -B build-vision \
        "-DCMAKE_MAKE_PROGRAM=${NINJA_BIN}" \
        -DCMAKE_BUILD_TYPE=Release -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON \
        -DGGML_CCACHE=OFF
    "$VENV/bin/cmake" --build build-vision --target strata-vision -j "$(nproc)"
fi

if [ "$STRATA_SPLIT" = "1" ]; then
    "$VENV/bin/cmake" -G Ninja -S engine -B build-split \
        "-DCMAKE_MAKE_PROGRAM=${NINJA_BIN}" \
        -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
        -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_SPLIT=ON -DSTRATA_PORTABLE=ON \
        -DGGML_CCACHE=OFF \
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

# --- stage the main engine and Inkling as signed-feed-compatible component
#     packages. The runtime resolves component:strata-engine and
#     component:strata-inkling through STRATA_COMPONENTS_DIR. ---
cd "$SRC"
version=$("$VENV/bin/python" -c 'import json; print(json.load(open("terminal/engine/BUILD.json"))["version"])')
cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
cudamajor=${cudafull%%.*}
if [ "$cudamajor" = "13" ]; then
    mindriver=580.65
else
    mindriver=575.51
fi
arch_args=()
for a in $(echo "${STRATA_ARCH}" | tr ';,' '  '); do arch_args+=(--arch "sm_${a}"); done
arch_args+=(--arch "compute_${last}")
build_id="infer-images-${STRATA_REF}-cuda${cudafull}"
components=/install/strata/components
stage_root=/tmp/strata-component-stage
rm -rf "$components" "$stage_root"
mkdir -p "$components/engine" "$components/shapes/inkling2/engine" "$stage_root"

package_component() {
    local name="$1" shape="$2" binary="$3" stage="$4"
    mkdir -p "$stage"
    cp "$binary" "$stage/$name"
    cp "$SRC/terminal/engine/BUILD.json" "$stage/BUILD.json"
    cp "$SRC/LICENSE" "$stage/LICENSE"
    if [ "$name" = "strata" ]; then
        if [ -f "$SRC/terminal/engine/libstrata_kq.so" ]; then
            cp "$SRC/terminal/engine/libstrata_kq.so" "$stage/"
        fi
        if [ -f "$SRC/terminal/engine/strata-vision" ]; then
            cp "$SRC/terminal/engine/strata-vision" "$stage/"
        fi
    fi
    "$VENV/bin/python" "$SRC/tools/packaging/collect_libs.py" \
        --binary "$stage/$name" --search /usr/local/cuda/lib64 --out "$stage" \
        --also 'libcublasLt.so*'
    local package_args=(
        --src "$stage" --backend cuda --version "$version"
        --platform linux-x86_64 --variant "cuda${cudamajor}"
        --min-driver "nvidia=${mindriver}" --build-id "$build_id"
        --include SYSTEM-LIBS.txt --include 'licenses/*'
        "${arch_args[@]}"
    )
    if [ -n "$shape" ]; then
        package_args+=(--shape "$shape")
    fi
    "$VENV/bin/python" "$SRC/tools/packaging/package_engine.py" "${package_args[@]}" --out "$5"
}

ink_bin=""
if [ -f "$SRC/build/src/inkling2/strata-inkling" ]; then
    ink_bin="$SRC/build/src/inkling2/strata-inkling"
elif [ -f "$SRC/build/strata-inkling" ]; then
    ink_bin="$SRC/build/strata-inkling"
fi
[ -n "$ink_bin" ] || { echo "ERROR: strata-inkling was not built" >&2; exit 1; }

package_component strata "" "$SRC/build/strata" "$stage_root/main" "$components/engine"
package_component strata-inkling inkling2 "$ink_bin" "$stage_root/inkling" "$components/shapes/inkling2/engine"

for package_dir in "$components"/engine/*/ "$components"/shapes/inkling2/engine/*/; do
    [ -d "$package_dir" ] || continue
    "$VENV/bin/python" "$SRC/tools/packaging/package_engine.py" --verify "$package_dir"
    "$VENV/bin/python" "$SRC/tools/packaging/binary_deps.py" "$package_dir"
done
rm -rf "$stage_root"

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
