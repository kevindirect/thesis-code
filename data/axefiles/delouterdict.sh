#!/bin/sh

# Deletes the old outer dictionary structure of axefiles
sed -i --follow-symlinks '2d' "$@";		# Delete second line
sed -i --follow-symlinks 'N;$!P;D' "$@";	# Delete second-to-last line
sed -i --follow-symlinks 's/	//' "$@";	# Delete first tab frome each line
