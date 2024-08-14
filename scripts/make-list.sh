#!/usr/bin/env bash

grep -hEr '^class\s.+(IStrategy)' $1 | cut -d' ' -f2 | cut -d'(' -f1 | xargs echo -n
