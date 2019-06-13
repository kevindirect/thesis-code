#!/bin/sh

sed -i --follow-symlinks '/^[[:space:]]*$/d' "$@";
sed -i --follow-symlinks -E -n 'H; x; s:,(\s*\n\s*}):\1:; P; ${x; p}' "$@";
sed -i --follow-symlinks '/^[[:space:]]*$/d' "$@";
