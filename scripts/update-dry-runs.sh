#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR/..

mkdir -p dry-runs

for d in $(find strategies/*/dry-run/* -type d); do
  strategy="$(echo $d | cut -d'/' -f2)"
  leaf="$(echo $d | cut -d'/' -f4-)"
  link_name="${leaf/\//_}_${strategy}"
  #echo "dry-runs/${link_name} -> ${d}"
  ln -Tfsrv "$d" "dry-runs/$link_name"
done
