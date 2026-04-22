#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist/macos"
APP_NAME="JForge"
APP_VERSION="${1:-1.0.0}"
MAIN_CLASS="atri.palaash.jforge.app.JForgeApp"
ICON_PATH="$ROOT_DIR/assets/icons/jforge.icns"

pushd "$ROOT_DIR" >/dev/null

if [[ ! -f app/target/jforge-app-1.0.0-SNAPSHOT.jar ]]; then
  mvn clean -DskipTests package
fi

mkdir -p "$DIST_DIR"

JPACKAGE_ARGS=(
  --type app-image
  --name "$APP_NAME"
  --dest "$DIST_DIR"
  --input app/target
  --main-jar jforge-app-1.0.0-SNAPSHOT.jar
  --main-class "$MAIN_CLASS"
  --app-version "$APP_VERSION"
  --vendor "JForge"
  --copyright "2026 JForge"
  --mac-package-name "JForge"
  --mac-package-identifier "atri.palaash.jforge"
)

if [[ -f "$ICON_PATH" ]]; then
  JPACKAGE_ARGS+=(--icon "$ICON_PATH")
fi

# Optional signing identity for Gatekeeper-friendly distribution.
if [[ -n "${JFORGE_MAC_SIGN_IDENTITY:-}" ]]; then
  JPACKAGE_ARGS+=(
    --mac-sign
    --mac-signing-key-user-name "$JFORGE_MAC_SIGN_IDENTITY"
  )
fi

jpackage "${JPACKAGE_ARGS[@]}"

popd >/dev/null

echo "macOS app bundle created at: $DIST_DIR/$APP_NAME.app"
