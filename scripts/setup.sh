#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/lib.sh"

usage() {
	echo "Usage: $BIN_NAME [ -i | -u ]

Manage freqtrade instance

Arguments:
  -i            Install freqtrade
  -u            Update freqtrade
  -h, --help    Display this help and exit"
}

install() {
  echo "===> install"
  test -e "$FREQTRADE_DIR" && \
    echo "===> already installed"; \
    exit
  git clone --branch develop git@github.com:freqtrade/freqtrade.git "$FREQTRADE_DIR"
}

update() {
  echo "===> update"
  cd "$FREQTRADE_DIR"
  ./setup.sh -u
}

ACTION=install

ARGS=$(getopt -o 'hiu' --long 'help' -- "$@")
eval set -- "$ARGS"
while true; do
  case "$1" in
    -i)
      ACTION=install
      shift 1
      continue
      ;;
    -u)
      ACTION=update
      shift 1
      continue
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    '--')
      shift
      break
      ;;
    *)
      echo 'Internal error'
      exit 1
      ;;
  esac
done

case "$ACTION" in
  install)
    install
    ;;
  update)
    update
    ;;
  *)
    echo "Invalid action: $ACTION"
    usage
    exit 1
    ;;
esac

echo "===> done"
