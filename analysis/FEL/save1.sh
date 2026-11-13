#!/bin/bash

input_file="$1"
output_file="${2:-$input_file}"

temp_file=$(mktemp) || exit 1

perl -ne '
    if ($_ =~ /^\s*?\S+(\s+\S+)/) {
        print "$1\n";
    } else {
        print "\n";
    }
' "$input_file" > "$temp_file"

mv "$temp_file" "$output_file" && \
echo "Done. Output file: $output_file"
