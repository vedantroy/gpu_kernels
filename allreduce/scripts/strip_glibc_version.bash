#!/bin/bash
set -euo pipefail

# Check if the file path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_so_file>"
    exit 1
fi

# Assign the file path to a variable
so_file="$1"
to_remove="GLIBC_2.32"

# Check if the file exists
if [ ! -f "$so_file" ]; then
    echo "Error: File '$so_file' not found."
    exit 2
fi

# Print the glibc versions
versions=$(nm --dynamic --undefined-only --with-symbol-versions "$so_file" \
  | grep GLIBC | sed -e 's#.\+@##' | sort --unique)
echo "$versions"

# Extract the symbols that use GLIBC_2.29 and clear their version
nm --dynamic --undefined-only --with-symbol-versions "$so_file" | grep ${to_remove} | awk '{print $3}' | \
while read -r symbol; do
    echo "Clearing version for symbol: $symbol"
    patchelf --clear-symbol-version "$symbol" "$so_file"
done

echo "FINISHED"
# Print new versions
versions=$(nm --dynamic --undefined-only --with-symbol-versions "$so_file" \
  | grep GLIBC | sed -e 's#.\+@##' | sort --unique)
echo "$versions"