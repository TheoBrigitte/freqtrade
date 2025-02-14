#!/usr/bin/env bash
#
# Setup script to install and update freqtrade

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/common.sh"

usage() {
  echo "Usage: $(basename $0) [ -i | -u ]

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
    return
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

# Install additional tools
install_tools

# Replace freqtrade's backtest results with symlink to this repository backtest_results
test -d "$FREQTRADE_DIR/user_data/backtest_results" && rm -r "$FREQTRADE_DIR/user_data/backtest_results"
ln -fTsrv "$SCRIPT_DIR/../backtest_results" "$FREQTRADE_DIR/user_data/backtest_results"

echo "===> done"
