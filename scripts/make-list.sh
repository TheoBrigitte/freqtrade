#!/usr/bin/env bash
#
# Make a list of strategy names from a directory

if [ $# -ne 1 ]; then
  echo "Usage: $(basename $0) <startegy directory or file>

This script make a space separated list of strategy names from a directory or file."
  exit 1
fi

grep -hEr '^class\s.+(IStrategy)' $1 | cut -d' ' -f2 | cut -d'(' -f1 | xargs echo -n
