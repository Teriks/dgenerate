#!/bin/sh
# dgenerate installer bootstrap (Linux / macOS Apple Silicon)
#
# Usage:
#   curl -fsSL https://github.com/Teriks/dgenerate/releases/latest/download/install.sh | sh
#
# Windows (PowerShell):
#   irm https://github.com/Teriks/dgenerate/releases/latest/download/install.ps1 | iex
#
# Downloads the installer attached to this GitHub release and launches it.
# In the installer UI, choose "Latest Release" for the current stable build.
#
# The release workflow replaces the tag placeholder on the TAG= line below
# before uploading this file. Do not repeat that placeholder earlier in the file.

set -eu

REPO="Teriks/dgenerate"
TAG="@DGENERATE_RELEASE_TAG@"
API_URL="https://api.github.com/repos/${REPO}/releases/tags/${TAG}"

say()  { printf '%s\n' "$*"; }
err()  { printf 'error: %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

case "$TAG" in
    @DGENERATE_RELEASE_TAG@)
        die "this installer script was not published with a release tag."
        ;;
esac

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
# Asset names include the version (e.g. dgenerate-installer-linux-5.0.0),
# so look the current name up from the release metadata.
say "Looking up the ${TAG} installer..."
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
say "  -> Select \"Latest Release\" to install the current stable version."
say ""

# stdin is the pipe from curl, so reattach the terminal for the interactive installer.
if [ -r /dev/tty ]; then
    "$INSTALLER" "$@" </dev/tty
else
    "$INSTALLER" "$@"
fi
