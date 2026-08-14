#!/bin/sh

# Linux installer for MamboRambo
# Supports Debian / Ubuntu with DEB packages
# Accepts tag in the first argument

# Usage:
# ./installer.sh {tag}

# Available at https://maxmelichov.github.io/MamboRambo/installer.sh
# Via curl -sSf https://maxmelichov.github.io/MamboRambo/installer.sh | sh -s {tag}

set -e

TAG=$1
VERSION=$(echo "$TAG" | sed 's/^mamborambo-desktop-v//; s/^v//')

if [ -z "$TAG" ]; then
    echo "Error: No tag specified. Usage: ./installer.sh {tag}"
    exit 1
fi

ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

DEB_ARCH="amd64"

DEB_URL="https://github.com/maxmelichov/MamboRambo/releases/download/${TAG}/mamborambo_${VERSION}_${DEB_ARCH}.deb"

echo "Downloading MamboRambo version $TAG for $ARCH..."

TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

if [ -f /etc/os-release ] && grep -iq "ubuntu\|debian" /etc/os-release; then
    echo "Detected Debian/Ubuntu. Downloading DEB package..."
    wget "$DEB_URL" -O mamborambo.deb
    sudo apt-get install -y ./mamborambo.deb
else
    echo "Unsupported Linux distribution. Please install the DEB package manually:"
    echo "$DEB_URL"
    exit 1
fi

cd ..
rm -rf "$TEMP_DIR"

echo "MamboRambo installation complete!"
echo "Run 'mamborambo' to open it!"
