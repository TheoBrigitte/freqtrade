#!/usr/bin/env bash

FREQTRADE_DIR="$SCRIPT_DIR/../freqtrade"

args=()

if [ "$FREQTRADE_MODE" == "futures" ]; then
  args+=(--config ./config/config_futures.json)
  args+=(--config ./pairlist/binance-usdt-static_futures.json)
else
  args+=(--config ./config/config_spot.json)
  args+=(--config ./pairlist/binance-usdt-static_spot.json)
fi

#args+=(--config ./config/config_fix10100.json)
args+=(--config ./config/config_3unlimited.json)
args+=(--user-data-dir $FREQTRADE_DIR/user_data)
