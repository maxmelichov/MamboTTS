#!/usr/bin/env bash
# Build MamboTTS desktop Linux bundles inside Docker (x86_64).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="mambotts-linux-builder:22.04"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building cached Linux builder image (one-time)..."
  docker build --platform linux/amd64 -t "$IMAGE" -f "$ROOT/scripts/Dockerfile.linux-build" "$ROOT/scripts"
fi

docker run --rm \
  --platform linux/amd64 \
  -v "$ROOT:/workspace" \
  -v mambotts-desktop-node-modules:/workspace/mambotts-desktop/node_modules \
  -w /workspace \
  -e CI=true \
  -e ORT_STRATEGY=system \
  -e ORT_LIB_LOCATION=/workspace/crates/blue-rs/.ort/onnxruntime-linux-x64-1.23.2/lib \
  -e ORT_PREFER_DYNAMIC_LINK=1 \
  -e LD_LIBRARY_PATH=/workspace/crates/blue-rs/.ort/onnxruntime-linux-x64-1.23.2/lib \
  -e LIBRARY_PATH=/workspace/crates/blue-rs/.ort/onnxruntime-linux-x64-1.23.2/lib \
  -e RUSTFLAGS="-L native=/workspace/crates/blue-rs/.ort/onnxruntime-linux-x64-1.23.2/lib" \
  -e APPIMAGE_EXTRACT_AND_RUN=1 \
  -e NO_STRIP=true \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    export PATH="/root/.cargo/bin:/root/.local/bin:$PATH"
    export CI=true
    uv run scripts/pre_build.py --target x86_64-unknown-linux-gnu
    pnpm --dir mambotts-desktop install --frozen-lockfile
    pnpm --dir mambotts-desktop exec tauri build --target x86_64-unknown-linux-gnu
  '

echo "Linux bundles:"
find "$ROOT/target/x86_64-unknown-linux-gnu/release/bundle" -type f \( -name "*.AppImage" -o -name "*.deb" -o -name "*.rpm" \) -print 2>/dev/null || true
