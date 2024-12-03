#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/common.sh"

timeframe="1m 3m 5m 15m 30m 1h 4h 12h 1d"

if [ $# -ne 1 ]; then
  echo "Usage: $(basename $0) <timerange>

This script will download data for the $timeframe timeframes and the given timerange.

timerange format: YYYYMMDD-YYYYMMDD, e.g. 20230301-20230601"
  exit 1
fi

$FREQTRADE_DIR/.venv/bin/freqtrade download-data ${args[@]} --timerange "$1" --timeframe $timeframe

echo "==> done"
