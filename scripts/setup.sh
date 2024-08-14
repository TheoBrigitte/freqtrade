#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/lib.sh"

action="${1--i}"
shift

if [ "$action" == "-i" ]; then
  echo "===> install"
  test -e "$FREQTRADE_DIR" && \
    echo "===> already installed"; \
    exit
  git clone --branch develop git@github.com:freqtrade/freqtrade.git "$FREQTRADE_DIR"
fi

if [ "$action" == "-u" ]; then
  echo "===> update"
  cd "$FREQTRADE_DIR"
  ./setup.sh -u
fi

echo "===> done"
