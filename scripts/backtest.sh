#!/usr/bin/env bash

set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/lib.sh"

input="$1"
shift

source_directory=""
strategy_list=""
if [ -d "$input" ]; then
  source_directory="$input"
  #$SCRIPT_DIR/migrate-strategies.sh
  if [ -f "$source_directory/whitelist" ]; then
    strategy_list=$(cat "$source_directory/whitelist")
  else
    strategy_list=$($SCRIPT_DIR/make-list.sh "$source_directory")
  fi
else
  source_directory="$(dirname "$input")"
  strategy_list=$($SCRIPT_DIR/make-list.sh "$input")
fi

strategy_count=$(echo "$strategy_list" | wc -w)

echo "==> strategy directory: $source_directory"
echo "==> strategy list ($strategy_count) : $strategy_list"

#args+=(--timerange 20230301-20230601)

#args+=(--recursive-strategy-search)
args+=(--strategy-path $source_directory)

if [ -f "$source_directory/config.json" ]; then
	#echo "==> found timeframes:"
	##grep -hEr 'timeframe =' $target_directory | awk '{print $3}' | sort -bi | uniq -c
	#grep -hPr '^\s+timeframe\s+=\s+.\d+[mhd].' $target_directory | awk '{print $3}' | sort -bi | uniq -c
	#read -p "Enter timeframe: " timeframe
	#args+=(--timeframe $timeframe)
	args+=(--config "$source_directory/config.json")
fi

if [ -f "$source_directory/arguments" ]; then
	args+=($(cat "$source_directory/arguments"))
fi

echo "==> backtest start"
(
  set -x
  $FREQTRADE_DIR/.venv/bin/freqtrade backtesting --cache none --enable-protections --timeframe-detail 1m "${args[@]}" --strategy-list $strategy_list $@
  #$FREQTRADE_DIR/.venv/bin/freqtrade backtesting --cache none --disable-max-market-positions --timeframe-detail 1m "${args[@]}" --strategy-list $strategy_list $@
)

echo "===> profit plot command"
echo $FREQTRADE_DIR/.venv/bin/freqtrade plot-profit "${args[@]}" $@ --strategy strategy_name
if [ $strategy_count -eq 1 ]; then
  echo "==> plotting profit start"
  (
    set -x
    $FREQTRADE_DIR/.venv/bin/freqtrade plot-profit "${args[@]}" $@ --strategy $strategy_list
  )
fi