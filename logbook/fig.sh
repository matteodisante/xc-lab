#!/usr/bin/env bash
# Freeze a figure for the logbook: logbook/fig.sh FILE NAME [DATE] writes fig/DATE_NAME.png
# (DATE defaults to today), so later runs that overwrite out/ do not change the entry.
set -euo pipefail
src=$1 name=$2 day=${3:-$(date +%F)}
dst="$(dirname "$0")/fig/${day}_${name}"
mkdir -p "$(dirname "$dst")"
case "$src" in
  *.pdf) pdftoppm -png -r 110 -singlefile "$src" "$dst" ;;
  *.png) cp "$src" "$dst.png" ;;
  *) echo "PDF or PNG only" >&2; exit 1 ;;
esac
echo "![]($(basename "$dst").png)" | sed 's|^!\[\](|![](fig/|'
