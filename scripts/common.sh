#!/usr/bin/env bash

FREQTRADE_DIR="$SCRIPT_DIR/../freqtrade"

get_args() {
  args=()

  args+=(--config ./config/config_base.json)
  #args+=(--config ./config/config_baseline.json)

  if [ "$FREQTRADE_MODE" == "futures" ]; then
    args+=(--config ./config/config_futures.json)

    args+=(--config ./pairlist/binance-usdt-static_futures.json)
    #args+=(--config ./pairlist/monthly_60_USDT_0,0_minprice_current.json)
  else
    args+=(--config ./pairlist/binance-usdt-static_spot.json)
    #args+=(--config ./pairlist/monthly_60_USDT_0,05_minprice_current_1.json)

    #args+=(--config ./config/config_spot_usdc.json)
    #args+=(--config ./pairlist/hyperliquid-usdc-static_spot.json)
  fi

  args+=(--config ./config/config_blacklist_stablecoins.json)
  #args+=(--config ./config/config_fix10100.json)
  args+=(--config ./config/config_3unlimited.json)

  args+=(--user-data-dir $FREQTRADE_DIR/user_data)

  echo "${args[@]}"
}
