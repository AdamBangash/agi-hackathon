#!/usr/bin/env bash
# Starts the OpenAI-compatible Gemma server on :8000 in the background.
SNAP=$(ls -d ~/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it-qat-q4_0-gguf/snapshots/*/ | head -1)
pkill -f "llama-server.*--port 8000" || true
nohup ~/llama.cpp/build/bin/llama-server \
  -m "$SNAP/gemma-4-26B_q4_0-it.gguf" --mmproj "$SNAP/gemma-4-26B-it-mmproj.gguf" \
  --alias gemma-4-26b-a4b-it -ngl 999 -c 16384 --jinja \
  --host 0.0.0.0 --port 8000 > ~/gemma_server.log 2>&1 &
echo "started pid $!; log: ~/gemma_server.log"
