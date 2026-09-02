#!/bin/sh
# Launch the CD image in DuckStation (invoke the binary directly; `open -a`
# may be ignored by an already-running instance).
set -eu
DIR=$(cd "$(dirname "$0")" && pwd)
DS=${DUCKSTATION:-$HOME/Applications/DuckStation.app/Contents/MacOS/DuckStation}
exec "$DS" "$@" "$DIR/build/nullfighter.cue"
