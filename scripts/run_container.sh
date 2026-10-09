#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -t physical-ai-final .
docker run --rm -it --shm-size=2g -v "$PWD:/workspace" physical-ai-final
