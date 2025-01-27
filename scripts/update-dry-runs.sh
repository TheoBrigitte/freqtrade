#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR/..

cat <<EOF > dry-runs/README.md
This directory holds a list of dry runs that have been performed on various strategies.

#### dry-runs

EOF

mkdir -p dry-runs

for d in $(find strategies/*/dry-run/* -type d|sort -rt/ -k4); do
  strategy="$(echo $d | cut -d'/' -f2)"
  leaf="$(echo $d | cut -d'/' -f4-)"
  link_name="${leaf/\//_}_${strategy}"
  relative_path=$(realpath --relative-to=dry-runs "$d")
  echo "- [$link_name]($relative_path)" >> dry-runs/README.md
  ln -Tfsrv "$d" "dry-runs/$link_name"
done
