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

    # Tauri fetches linuxdeploy and its AppImage plugin into this cache the
    # first time it bundles an AppImage, and it reuses whatever is already
    # there. The plugin ships as a static-pie executable, which Rosetta cannot
    # run, so on an Apple Silicon Mac the AppImage step dies with an opaque
    # "subprocess failed" from linuxdeploy. Seed the cache ourselves and, when
    # the plugin turns out not to be executable here, put a wrapper in its
    # place that runs it under qemu-user instead. On a native x86_64 host the
    # plugin runs directly and the wrapper is never written.
    # The real plugin has to live outside the cache directory, because
    # linuxdeploy treats every "linuxdeploy-plugin-*" file it finds there as a
    # plugin and tries to execute it while it scans.
    cache="$HOME/.cache/tauri"
    plugin="$cache/linuxdeploy-plugin-appimage.AppImage"
    real="$HOME/.cache/mambotts/appimage-plugin"
    mkdir -p "$cache" "$(dirname "$real")"
    if [ ! -x "$real" ]; then
      curl -fsSL -o "$real" https://github.com/linuxdeploy/linuxdeploy-plugin-appimage/releases/download/continuous/linuxdeploy-plugin-appimage-x86_64.AppImage
      chmod +x "$real"
    fi
    if "$real" --plugin-api-version >/dev/null 2>&1; then
      cp "$real" "$plugin"
    else
      echo "linuxdeploy AppImage plugin is not directly executable here; running it under qemu-user"
      if ! command -v qemu-x86_64-static >/dev/null 2>&1; then
        apt-get update -qq
        apt-get install -y -qq qemu-user-static
      fi
      printf "#!/bin/bash\nexec /usr/bin/qemu-x86_64-static %s \"\$@\"\n" "$real" > "$plugin"
    fi
    chmod +x "$plugin"

    uv run scripts/pre_build.py --target x86_64-unknown-linux-gnu
    pnpm --dir mambotts-desktop install --frozen-lockfile
    pnpm --dir mambotts-desktop exec tauri build --target x86_64-unknown-linux-gnu
  '

echo "Linux bundles:"
find "$ROOT/target/x86_64-unknown-linux-gnu/release/bundle" -type f \( -name "*.AppImage" -o -name "*.deb" -o -name "*.rpm" \) -print 2>/dev/null || true
