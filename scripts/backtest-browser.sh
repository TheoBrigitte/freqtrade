#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PATH="$SCRIPT_DIR:$PATH"

source "$SCRIPT_DIR/common.sh"

install_tools

freqtrade_args=($(get_args))
freqtrade_cmd="$FREQTRADE_DIR/.venv/bin/freqtrade backtesting-show ${freqtrade_args[@]} --export-filename"
backtest_results_dir="$SCRIPT_DIR/../backtest_results"

ls -t $backtest_results_dir/backtest-result-*.json | \
  grep -v ".meta.json" | \
  xargs basename -a | \
  $FZF_BIN \
    --preview="$JQ_BIN -r '[.strategy[]|(.exit_reason_summary[]|select(.key == \"TOTAL\"))+{\"key\":.strategy_name}+pick(.expectancy,.expectancy_ratio,.profit_factor,.max_relative_drawdown)]' $backtest_results_dir/{}" \
    --bind "enter:execute($freqtrade_cmd $backtest_results_dir/{} | less +G)" \
    --bind alt-enter:accept \
    --header="enter:show details, alt-enter:select entry and exit"
