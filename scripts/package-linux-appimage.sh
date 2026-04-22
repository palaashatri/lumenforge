#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist/linux"
APP_NAME="JForge"
APP_VERSION="${1:-1.0.0}"
MAIN_CLASS="atri.palaash.jforge.app.JForgeApp"
ICON_PATH="$ROOT_DIR/assets/icons/jforge.png"

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
  --linux-package-name "jforge"
  --linux-shortcut
)

if [[ -f "$ICON_PATH" ]]; then
  JPACKAGE_ARGS+=(--icon "$ICON_PATH")
fi

jpackage "${JPACKAGE_ARGS[@]}"

if command -v appimagetool >/dev/null 2>&1; then
  appimagetool "$DIST_DIR/$APP_NAME" "$DIST_DIR/$APP_NAME-$APP_VERSION.AppImage"
  echo "Linux AppImage created at: $DIST_DIR/$APP_NAME-$APP_VERSION.AppImage"
else
  echo "appimagetool is not installed. App image folder created at: $DIST_DIR/$APP_NAME"
fi

popd >/dev/null
