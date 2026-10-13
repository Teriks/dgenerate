#!/bin/sh
# dgenerate pre-release installer bootstrap (Linux / macOS Apple Silicon)
#
# Usage:
#   curl -fsSL https://github.com/Teriks/dgenerate/releases/download/pre-release/install.sh | sh
#
# Windows (PowerShell):
#   irm https://github.com/Teriks/dgenerate/releases/download/pre-release/install.ps1 | iex
#
# Downloads the latest installer from the "pre-release" GitHub release
# and launches it. In the installer UI, choose "Development Branch" and
# select the version_6.0.0 option to install the pre-release build.

set -eu

REPO="Teriks/dgenerate"
TAG="pre-release"
API_URL="https://api.github.com/repos/${REPO}/releases/tags/${TAG}"

say()  { printf '%s\n' "$*"; }
err()  { printf 'error: %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- pick a downloader -------------------------------------------------------
if have curl; then
    fetch()      { curl -fsSL "$1"; }
    fetch_file() { curl -fL --progress-bar -o "$2" "$1"; }
elif have wget; then
    fetch()      { wget -qO- "$1"; }
    fetch_file() { wget -O "$2" "$1"; }
else
    die "curl or wget is required."
fi

# --- detect platform ---------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        PLATFORM="linux"
        case "$ARCH" in
            x86_64|amd64) ;;
            *) err "warning: the Linux installer is built for x86_64; detected '$ARCH'. Continuing anyway." ;;
        esac
        ;;
    Darwin)
        PLATFORM="macos"
        case "$ARCH" in
            arm64|aarch64) ;;
            *) die "the macOS installer supports Apple Silicon (arm64) only; detected '$ARCH'." ;;
        esac
        ;;
    *)
        die "unsupported OS '$OS'. On Windows PowerShell: irm https://github.com/${REPO}/releases/download/${TAG}/install.ps1 | iex"
        ;;
esac

# --- find the installer asset ------------------------------------------------
# Asset names include the commit hash (e.g. dgenerate-installer-linux-006ec0b),
# so look the current name up from the release metadata.
say "Looking up the latest ${TAG} build..."
RELEASE_JSON="$(fetch "$API_URL")" || die "could not fetch release info from ${API_URL} (GitHub API rate limit?)"

ASSET_URL="$(printf '%s\n' "$RELEASE_JSON" \
    | grep -o '"browser_download_url"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | sed 's/.*"\(https[^"]*\)"$/\1/' \
    | grep "/dgenerate-installer-${PLATFORM}-" \
    | head -n 1 || true)"

[ -n "$ASSET_URL" ] || die "no ${PLATFORM} installer found in release '${TAG}'. See https://github.com/${REPO}/releases/tag/${TAG}"

ASSET_NAME="${ASSET_URL##*/}"

# --- download ----------------------------------------------------------------
TMPDIR_INSTALL="$(mktemp -d 2>/dev/null || mktemp -d -t dgenerate)"
trap 'rm -rf "$TMPDIR_INSTALL"' EXIT INT TERM HUP

INSTALLER="${TMPDIR_INSTALL}/${ASSET_NAME}"

say "Downloading ${ASSET_NAME}..."
fetch_file "$ASSET_URL" "$INSTALLER" || die "download failed."

chmod +x "$INSTALLER"

# macOS: strip the quarantine attribute so the binary can run
if [ "$PLATFORM" = "macos" ] && have xattr; then
    xattr -d com.apple.quarantine "$INSTALLER" 2>/dev/null || true
fi

# --- run ---------------------------------------------------------------------
say ""
say "Launching the installer."
say "  -> Select \"Development Branch\" and choose version_6.0.0 to install the pre-release."
say ""

# stdin is the pipe from curl, so reattach the terminal for the interactive installer.
if [ -r /dev/tty ]; then
    "$INSTALLER" "$@" </dev/tty
else
    "$INSTALLER" "$@"
fi
