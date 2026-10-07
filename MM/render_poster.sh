#!/usr/bin/env bash
# Rebuild poster.pdf whenever a source, style, bibliography, or figure changes.
# Usage: ./render_poster.sh | ./render_poster.sh --once | ./render_poster.sh --clean

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Project-local paper font; do not alter the system TeX installation.
export TEXINPUTS="$SCRIPT_DIR/vendor/libertine/latex//:${TEXINPUTS:-}"
export TFMFONTS="$SCRIPT_DIR/vendor/libertine/tfm//:${TFMFONTS:-}"
export VFFONTS="$SCRIPT_DIR/vendor/libertine/vf//:${VFFONTS:-}"
export T1FONTS="$SCRIPT_DIR/vendor/libertine/type1//:${T1FONTS:-}"
export ENCFONTS="$SCRIPT_DIR/vendor/libertine/enc//:${ENCFONTS:-}"
export TEXFONTMAPS="$SCRIPT_DIR/vendor/libertine/map//:${TEXFONTMAPS:-}"

BUILD_DIR="$SCRIPT_DIR/build"
MAIN_TEX="poster.tex"
OUTPUT_PDF="$SCRIPT_DIR/poster.pdf"
MODE="${1:---watch}"

clean() {
  rm -f -- "$BUILD_DIR"/poster.{aux,log,nav,out,pdf,snm,toc,vrb}
  printf 'Removed generated files from %s\n' "$BUILD_DIR"
}

source_signature() {
  find "$SCRIPT_DIR" -maxdepth 2 -type f \
    \( -name '*.tex' -o -name '*.sty' -o -name '*.bib' -o \
       -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.pdf' \) \
    ! -path "$BUILD_DIR/*" ! -path "$OUTPUT_PDF" \
    -printf '%T@ %s %p\n' | sort | sha256sum | cut -d' ' -f1
}

build() {
  mkdir -p "$BUILD_DIR"
  printf '\n[%(%H:%M:%S)T] Building %s ...\n' -1 "$MAIN_TEX"

  local pass status=0
  for pass in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
      -output-directory="$BUILD_DIR" "$MAIN_TEX" >"$BUILD_DIR/console.log" 2>&1 || {
        status=$?
        break
      }
  done

  if (( status == 0 )); then
    cp -f -- "$BUILD_DIR/poster.pdf" "$OUTPUT_PDF"
    local pages size
    pages="$(pdfinfo "$OUTPUT_PDF" 2>/dev/null | awk '/^Pages:/ {print $2}')"
    size="$(pdfinfo "$OUTPUT_PDF" 2>/dev/null | awk -F': *' '/^Page size:/ {print $2}')"
    printf '[%(%H:%M:%S)T] OK -> %s (%s page; %s)\n' -1 "$OUTPUT_PDF" "${pages:-?}" "${size:-unknown size}"
  else
    printf '[%(%H:%M:%S)T] Build failed. Last log lines:\n' -1 >&2
    tail -n 25 "$BUILD_DIR/console.log" >&2
  fi
  return "$status"
}

case "$MODE" in
  --once)
    build
    exit $?
    ;;
  --clean)
    clean
    exit 0
    ;;
  --watch|-w)
    ;;
  *)
    printf 'Usage: %s [--watch|-w|--once|--clean]\n' "${0##*/}" >&2
    exit 2
    ;;
esac

printf 'Watching %s for poster source changes. Press Ctrl+C to stop.\n' "$SCRIPT_DIR"
last_signature=""
while true; do
  current_signature="$(source_signature)"
  if [[ "$current_signature" != "$last_signature" ]]; then
    build || true
    last_signature="$current_signature"
  fi
  sleep 1
done
