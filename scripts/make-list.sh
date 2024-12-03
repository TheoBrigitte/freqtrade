#!/usr/bin/env bash
#
# Make a list of strategy names from a directory

grep -hEr '^class\s.+(IStrategy)' $1 | cut -d' ' -f2 | cut -d'(' -f1 | xargs echo -n
