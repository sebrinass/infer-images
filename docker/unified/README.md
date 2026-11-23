# Unified Docker Container

These scripts create a custom llama-swap container. Every component can be
left out of the build to shrink the image. The default set is llama.cpp +
llama-swap + vLLM + Strata; everything marked "opt-in" below is off
unless you ask for it:

- llama-server for LLMs, rerank and embedding model support (CUDA + Vulkan dual backend, runs on NVIDIA/AMD/Intel GPUs) — `--no-llama`
- llama.cpp's llama-tts tool (opt-in, ~128 MB) — `--llamatts`
- audiocpp_server for TTS and ASR (opt-in) — `--audiocpp` / `--no-audiocpp`
- vLLM for OpenAI-compatible high-throughput serving — `--no-vllm` (~4.7 GB)
- SGLang (isolated venv, opt-in: vLLM covers the same models) — `--sglang` / `--no-sglang` (~9.5 GB)
- Strata (Qwen3.8-Flash-Next engine) — `--no-strata` (~1 GB)
- Edge0 145B / 120B latent-MoE engine (`edge0-strata`, installed to
  `/opt/edge0`, separate from `/opt/strata`) — opt-in since 2026-11-16, `--edge0`
  / `--no-edge0` (~1 GB). Model packs are host mounts, not image content;
  launch with `/opt/edge0/run <pack>`. Dropped from the default set because the
  120B pack ran ~14 tok/s on 2x A2 and lost to Swift-Qwen3.8-27B on five of six
  published quality benchmarks.
- the llama-swap binary itself (the image entrypoint) — `--no-llama-swap`
- stable-diffusion.cpp (off by default, opt in with `--sd`)
- Hindsight's hs-decide-1-flash decision model (off by default, opt in with
  `--hsdecide`). A 0.8B single-forward classifier, not a generator, so it needs
  neither vLLM nor llama.cpp: the image vendors the `serving/` package (which
  adds `POST /v1/systemone`) and installs it, `--no-deps`, into `/opt/vllm-venv`
  so it runs on that venv's CUDA torch. Adds ~50 KB. The 1.5 GB weights are a
  host mount, not image content. On an A2 a decision with its candidates
  attached takes ~0.39 s, against ~31 s on the CPU cores; the file also ships a
  `hs-decide-serve` launcher. Requires `INCLUDE_VLLM=1`.
- the rootless variant, a second image that runs as an unprivileged user
  (off by default, opt in with `--rootless`; it re-chowns ~20 GB of venv
  files, which Docker copies into a full extra layer)

Two vLLM add-ons are opt-in as well (off by default):

- vLLM-Omni — `--vllm-omni`. Its version is auto-aligned to the vLLM
  major.minor line (vLLM 0.31.x -> vLLM-Omni 0.31.x, pre-releases included),
  because the two packages release on independent schedules and a stable
  omni that lags vLLM breaks with import errors. Override with `VLLM_OMNI_REF`.
- vllm-gguf-plugin (GGUF models served directly by vLLM) — `--vllm-gguf`

Example: `./build-image.sh --cuda --no-sglang --no-vllm` builds a ~3 GB image
with just llama.cpp + Strata. Full image: ~11 GB compressed.

Switches are also available as `INCLUDE_*` env vars (`INCLUDE_SGLANG=1
./build-image.sh --cuda`) and, for manually-triggered CI runs, as
workflow_dispatch checkboxes. Push/scheduled builds include the default set
and leave every opt-in component out (SGLang, audio.cpp, llama-tts, SD,
hs-decide, the vLLM add-ons, the rootless variant).

At build time the baked-in config has the entries of disabled components
commented out automatically (matched by what the entry's cmd invokes, not by
its name), so the image config never references a binary that was left out.
A bind-mounted config is NOT touched — keep it in sync manually.
