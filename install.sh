#!/bin/sh
# MamboTTS installer for macOS and Linux.
#
#   curl -fsSL https://github.com/maxmelichov/MamboTTS/releases/latest/download/install.sh | sh
#
# On macOS this downloads the DMG, copies MamboTTS.app into /Applications, and
# then removes the quarantine attribute. That last step matters: the macOS build
# is ad-hoc signed rather than Developer ID signed, so Gatekeeper refuses to open
# it on first launch unless you know the right-click Open trick. Stripping the
# quarantine flag from an app you just downloaded on purpose is the same decision
# that dialog is asking you to make, made once, in the open.
#
# On Linux this downloads the AppImage into ~/.local/bin, makes it executable,
# and writes a launcher entry so MamboTTS shows up in your applications menu.
# Nothing here needs sudo on Linux.
#
# The script is safe to run again. A second run replaces the previous install
# rather than leaving a second copy behind.

set -eu

REPO="maxmelichov/MamboTTS"
TAG_PREFIX="mambotts-desktop-"
# Used when the GitHub API cannot be reached, which happens often enough from
# shared or rate limited addresses that a hard failure would be the wrong
# default. Keep this in step with the newest published desktop release.
FALLBACK_VERSION="1.2.0"

APP_NAME="MamboTTS"
APPLICATIONS_DIR="${APPLICATIONS_DIR:-/Applications}"
LINUX_BIN_DIR="${HOME}/.local/bin"
LINUX_DESKTOP_DIR="${HOME}/.local/share/applications"
LINUX_ICON_DIR="${HOME}/.local/share/icons/hicolor/256x256/apps"

WORK_DIR=""

log() { printf '%s\n' "$*"; }
step() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

cleanup() {
    if [ -n "${WORK_DIR}" ] && [ -d "${WORK_DIR}" ]; then
        rm -rf "${WORK_DIR}"
    fi
}
trap cleanup EXIT INT TERM

require() {
    command -v "$1" >/dev/null 2>&1 || die "this installer needs \`$1\`, which is not on your PATH."
}

# Prints "os arch" for a combination we ship a build for, or exits with an
# explanation of what is missing.
detect_platform() {
    uname_s="$(uname -s)"
    uname_m="$(uname -m)"
    case "${uname_s}" in
        Darwin)
            case "${uname_m}" in
                arm64|aarch64) log "Detected macOS on Apple Silicon." ;;
                x86_64)
                    die "MamboTTS ships only an Apple Silicon build. This is an Intel Mac, and there is no x86_64 macOS bundle to install. Building from source is documented in docs/BUILDING.md."
                    ;;
                *) die "unrecognised macOS architecture \`${uname_m}\`." ;;
            esac
            PLATFORM="macos"
            ;;
        Linux)
            case "${uname_m}" in
                x86_64|amd64) log "Detected Linux on x86_64." ;;
                aarch64|arm64)
                    die "MamboTTS ships no Linux build for aarch64, only x86_64. Building from source is documented in docs/BUILDING.md."
                    ;;
                *) die "unrecognised Linux architecture \`${uname_m}\`." ;;
            esac
            PLATFORM="linux"
            ;;
        *)
            die "unsupported operating system \`${uname_s}\`. This installer covers macOS and Linux. Windows users should run install.ps1, which sets up the local MamboTTS server and its API."
            ;;
    esac
}

# Asks GitHub for the newest release whose tag starts with the desktop prefix.
# Falls back to the pinned version on any failure, including a rate limit,
# because being told an old version is available beats being told nothing.
resolve_version() {
    if [ -n "${MAMBOTTS_VERSION:-}" ]; then
        VERSION="${MAMBOTTS_VERSION}"
        log "Using MAMBOTTS_VERSION=${VERSION}."
        return
    fi

    step "Looking up the newest MamboTTS release"
    api_body=""
    api_body="$(curl -fsSL --max-time 20 \
        -H 'Accept: application/vnd.github+json' \
        "https://api.github.com/repos/${REPO}/releases?per_page=30" 2>/dev/null || true)"

    tag=""
    if [ -n "${api_body}" ]; then
        tag="$(printf '%s' "${api_body}" \
            | tr ',' '\n' \
            | grep '"tag_name"' \
            | sed -e 's/.*"tag_name"[[:space:]]*:[[:space:]]*"//' -e 's/".*//' \
            | grep "^${TAG_PREFIX}" \
            | head -n 1 || true)"
    fi

    if [ -n "${tag}" ]; then
        VERSION="$(printf '%s' "${tag}" | sed -e "s/^${TAG_PREFIX}//" -e 's/^v//')"
        log "Newest release is ${tag}."
    else
        VERSION="${FALLBACK_VERSION}"
        warn "could not read the GitHub releases API, which usually means a rate limit. Falling back to the pinned version ${VERSION}."
    fi

    [ -n "${VERSION}" ] || die "could not work out which version to install."
    RELEASE_BASE="https://github.com/${REPO}/releases/download/${TAG_PREFIX}v${VERSION}"
}

# Downloads a release asset, refusing anything that is not a real HTTP 200 with
# a non-empty body. Without this a GitHub error page happily installs itself as
# a zero byte AppImage.
download_asset() {
    asset="$1"
    dest="$2"
    url="${RELEASE_BASE}/${asset}"

    step "Downloading ${asset}"
    log "    from ${url}"
    status="$(curl -w '%{http_code}' -fL --retry 2 --connect-timeout 20 \
        -o "${dest}" "${url}" 2>/dev/null || true)"

    if [ "${status}" != "200" ]; then
        rm -f "${dest}"
        die "the download returned HTTP ${status:-000} instead of 200. Check that ${TAG_PREFIX}v${VERSION} still publishes ${asset}."
    fi
    [ -s "${dest}" ] || die "the downloaded ${asset} is empty."

    size="$(wc -c < "${dest}" | tr -d ' ')"
    if [ "${size}" -lt 1000000 ]; then
        die "the downloaded ${asset} is only ${size} bytes, which is far too small to be a real bundle."
    fi
    log "    ${size} bytes"
}

install_macos() {
    require hdiutil
    require ditto
    require xattr

    dmg="${WORK_DIR}/${APP_NAME}_${VERSION}_aarch64.dmg"
    download_asset "${APP_NAME}_${VERSION}_aarch64.dmg" "${dmg}"

    step "Verifying the disk image"
    hdiutil imageinfo "${dmg}" >/dev/null 2>&1 \
        || die "the download is not a readable macOS disk image."

    if [ ! -w "${APPLICATIONS_DIR}" ]; then
        die "${APPLICATIONS_DIR} is not writable by this user. Re-run the installer with sudo, or set APPLICATIONS_DIR to somewhere you own, for example: APPLICATIONS_DIR=\"\$HOME/Applications\" sh install.sh"
    fi

    mount_point="${WORK_DIR}/mnt"
    mkdir -p "${mount_point}"

    step "Mounting the disk image"
    hdiutil attach "${dmg}" -nobrowse -quiet -mountpoint "${mount_point}" \
        || die "could not mount the disk image."

    source_app="${mount_point}/${APP_NAME}.app"
    if [ ! -d "${source_app}" ]; then
        hdiutil detach "${mount_point}" -quiet || true
        die "the disk image does not contain ${APP_NAME}.app."
    fi

    target_app="${APPLICATIONS_DIR}/${APP_NAME}.app"
    if [ -d "${target_app}" ]; then
        step "Replacing the existing ${target_app}"
        rm -rf "${target_app}" || {
            hdiutil detach "${mount_point}" -quiet || true
            die "could not remove the existing ${target_app}."
        }
    else
        step "Installing to ${target_app}"
    fi

    # ditto rather than cp, so extended attributes and the bundle's symlinks
    # survive the copy intact.
    if ! ditto "${source_app}" "${target_app}"; then
        hdiutil detach "${mount_point}" -quiet || true
        die "could not copy ${APP_NAME}.app into ${APPLICATIONS_DIR}."
    fi

    step "Unmounting the disk image"
    hdiutil detach "${mount_point}" -quiet || warn "could not detach ${mount_point}. It is safe to eject it from Finder."

    step "Removing the quarantine attribute"
    log "    The macOS build is ad-hoc signed rather than Developer ID signed,"
    log "    so Gatekeeper would otherwise block the first launch and ask you to"
    log "    right-click and choose Open. Clearing com.apple.quarantine here is"
    log "    the same approval, granted once, to an app you asked for by name."
    xattr -dr com.apple.quarantine "${target_app}" 2>/dev/null || true

    if xattr -p com.apple.quarantine "${target_app}" >/dev/null 2>&1; then
        warn "the quarantine attribute is still set. Right-click ${APP_NAME}.app and choose Open for the first launch."
    else
        log "    Done. No quarantine attribute remains."
    fi

    log ""
    log "${APP_NAME} ${VERSION} is installed at ${target_app}."
    log "Open it from Launchpad, from Finder, or with: open -a ${APP_NAME}"
}

install_linux() {
    appimage_name="${APP_NAME}_${VERSION}_amd64.AppImage"
    staged="${WORK_DIR}/${appimage_name}"
    download_asset "${appimage_name}" "${staged}"

    step "Verifying the AppImage"
    magic="$(od -An -tx1 -N4 "${staged}" | tr -d ' \n')"
    if [ "${magic}" != "7f454c46" ]; then
        die "the download is not an ELF executable, so it is not a usable AppImage."
    fi

    mkdir -p "${LINUX_BIN_DIR}"
    target="${LINUX_BIN_DIR}/${APP_NAME}.AppImage"

    if [ -e "${target}" ]; then
        step "Replacing the existing ${target}"
    else
        step "Installing to ${target}"
    fi
    # Move onto the final name rather than versioning it, so repeat runs
    # replace the binary instead of accumulating copies.
    rm -f "${target}"
    mv "${staged}" "${target}"
    chmod 755 "${target}"

    step "Writing the launcher entry"
    mkdir -p "${LINUX_DESKTOP_DIR}"
    desktop_file="${LINUX_DESKTOP_DIR}/mambotts.desktop"

    icon_value="mambotts"
    if extract_linux_icon "${target}"; then
        log "    Icon extracted from the AppImage."
    else
        icon_value="audio-x-generic"
        log "    Could not read an icon out of the AppImage, so the entry uses a stock one."
    fi

    cat > "${desktop_file}" <<DESKTOP
[Desktop Entry]
Type=Application
Name=MamboTTS
GenericName=Text to Speech
Comment=Native offline BlueTTS for desktop
Exec=${target}
Icon=${icon_value}
Terminal=false
Categories=AudioVideo;Audio;Utility;
StartupWMClass=MamboTTS
DESKTOP
    chmod 644 "${desktop_file}"
    log "    ${desktop_file}"

    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "${LINUX_DESKTOP_DIR}" >/dev/null 2>&1 || true
    fi

    case ":${PATH}:" in
        *":${LINUX_BIN_DIR}:"*) : ;;
        *)
            warn "${LINUX_BIN_DIR} is not on your PATH, so typing \`${APP_NAME}.AppImage\` will not find it. Add this to your shell profile:"
            # shellcheck disable=SC2016  # this line is advice to paste, not a value to expand
            printf '\n    export PATH="$HOME/.local/bin:$PATH"\n\n' >&2
            ;;
    esac

    log ""
    log "${APP_NAME} ${VERSION} is installed at ${target}."
    log "Launch it from your applications menu, or run it directly:"
    log "    ${target}"
    log ""
    log "If it refuses to start with a FUSE error, your distribution is missing"
    log "libfuse2. Either install it, or run the AppImage with --appimage-extract-and-run."
}

# Best effort: an AppImage carries its icon as .DirIcon, and the runtime can
# unpack a single path without FUSE. A missing icon is cosmetic, so every
# failure here is non-fatal.
extract_linux_icon() {
    appimage="$1"
    extract_dir="${WORK_DIR}/icon"
    mkdir -p "${extract_dir}"
    (
        cd "${extract_dir}" || exit 1
        "${appimage}" --appimage-extract .DirIcon >/dev/null 2>&1
    ) || return 1

    for candidate in "${extract_dir}/squashfs-root/.DirIcon" "${extract_dir}/squashfs-root/mambotts.png"; do
        if [ -f "${candidate}" ]; then
            mkdir -p "${LINUX_ICON_DIR}"
            cp -f "${candidate}" "${LINUX_ICON_DIR}/mambotts.png" || return 1
            return 0
        fi
    done
    return 1
}

main() {
    require curl
    require uname

    log "MamboTTS installer"
    log ""
    detect_platform
    resolve_version

    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mambotts-install.XXXXXX")"

    case "${PLATFORM}" in
        macos) install_macos ;;
        linux) install_linux ;;
    esac
}

main "$@"
