#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PATH="$SCRIPT_DIR:$PATH"

source "$SCRIPT_DIR/common.sh"

freqtrade_args=($(get_args))
freqtrade_cmd="$FREQTRADE_DIR/.venv/bin/freqtrade backtesting-show ${freqtrade_args[@]} --export-filename"
backtest_results_dir="$SCRIPT_DIR/../backtest_results"

ls -t $backtest_results_dir/backtest-result-*.json | \
  grep -v ".meta.json" | \
  xargs basename -a | \
  $FZF_BIN --preview="$JQ_BIN -r '.strategy_comparison' $backtest_results_dir/{}" --bind "enter:execute($freqtrade_cmd $backtest_results_dir/{} | less +G)" --bind alt-enter:accept
