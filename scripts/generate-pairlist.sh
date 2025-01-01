#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/common.sh"

if [ $# -ne 1 ]; then
  echo "Usage: $(basename $0) <pairlist config>

This script will call test-pairlist and generate a pairlist.json file."
  exit 1
fi

args=($(get_args))

args+=(
  --config $1
)

pairs="$($FREQTRADE_DIR/.venv/bin/freqtrade test-pairlist ${args[@]} --print-json | jq -cr .)"
jq '.exchange.pair_whitelist='"$pairs" ./pairlist/binance-pairlist-template.json > pairlist.json
echo "==> generated: pairlist.json"
