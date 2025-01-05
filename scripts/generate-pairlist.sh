#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/common.sh"

if [ $# -ne 1 ] || [ "$1" == "-h" ]; then
  echo "Usage: $(basename $0) <pairlist config>

This script will call test-pairlist using the given pairlist handlers configuration and generate a pairlist.json file."
  exit 0
fi

args=($(get_args))

args+=(
  --config $1
)

pairs="$(freqtrade test-pairlist ${args[@]} --print-json | $JQ_BIN -cr .)"
$JQ_BIN '.exchange.pair_whitelist='"$pairs" "$SCRIPT_DIR/../pairlist/binance-pairlist-template.json" > pairlist.json
echo "==> generated: pairlist.json"
