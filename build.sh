#!/bin/sh
# Build on the remote machine that has the PSn00bSDK toolchain (mac-studio),
# then copy the resulting CD image back into ./build/.
#
#   ./build.sh            # rsync -> remote cmake/ninja -> rsync back
#   BUILD_HOST=foo ./build.sh
set -eu

HOST=${BUILD_HOST:-mac-studio.local}
DIR=$(cd "$(dirname "$0")" && pwd)
NAME=$(basename "$DIR")
REMOTE_DIR="work/github.com/GOROman/$NAME"

rsync -az --delete \
	--exclude .git --exclude build --exclude '*.bin' --exclude '*.cue' \
	"$DIR/" "$HOST:$REMOTE_DIR/"

ssh "$HOST" "zsh -lc 'cd $REMOTE_DIR && cmake --preset default . && cmake --build ./build'"

mkdir -p "$DIR/build"
rsync -az "$HOST:$REMOTE_DIR/build/nullfighter.bin" "$HOST:$REMOTE_DIR/build/nullfighter.cue" \
	"$HOST:$REMOTE_DIR/build/nullfighter.exe" "$HOST:$REMOTE_DIR/build/nullfighter.elf" "$DIR/build/"

echo
echo "=> $DIR/build/nullfighter.cue"
