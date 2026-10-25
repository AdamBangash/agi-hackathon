#!/usr/bin/env bash
# Sets up a Gemma 4 inference server on Jetson Thor (JetPack 7 / CUDA 13).
# Usage: SUDO_PASS=... ./thor_setup.sh
set -euo pipefail
S() { if [ -n "${SUDO_PASS:-}" ]; then echo "$SUDO_PASS" | sudo -S "$@"; else sudo "$@"; fi; }

S apt-get update -qq
S DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-2 cmake build-essential libcurl4-openssl-dev git

export PATH=/usr/local/cuda/bin:$PATH
cd ~
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build -j"$(nproc)" --target llama-server

# Model (skipped if already cached)
REPO=google/gemma-4-26B-A4B-it-qat-q4_0-gguf
[ -d ~/.cache/huggingface/hub/models--${REPO//\//--} ] || hf download "$REPO"
echo "setup done: ~/llama.cpp/build/bin/llama-server"
