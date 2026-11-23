# infer-images

Rolling Docker images for the local inference stack. This repository holds
build recipes only; every component is fetched at its latest version on each
build.

    docker pull ghcr.io/sebrinass/infer-images:latest

The image carries CUDA and Vulkan backends, llama.cpp, llama-swap, vLLM and
Strata by default; Edge0, SGLang, audio.cpp, hs-decide and the rest are opt-in.
Individual components are toggled with `INCLUDE_*` build args - see
`docker/unified/README.md`.

Builds run on pushes to `docker/unified/**` or the workflow, every 3 days, and
on manual dispatch. Not a fork; the llama-swap binary comes from its upstream
release.
