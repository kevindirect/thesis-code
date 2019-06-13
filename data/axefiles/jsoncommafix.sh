#!/bin/sh

# Fixes trailing commas in json files
sed -i --follow-symlinks '/^[[:space:]]*$/d' "$@";				# Remove empty lines
sed -i --follow-symlinks -E -n 'H; x; s:,(\s*\n\s*}):\1:; P; ${x; p}' "$@";	# Remove trailing comma
sed -i --follow-symlinks '/^[[:space:]]*$/d' "$@";				# Remove empty lines
