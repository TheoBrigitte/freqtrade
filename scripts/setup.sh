#!/usr/bin/env bash
#
# Author: Théo Brigitte
# Date: 2026-09-27
#
# Setup script to install and update freqtrade

# Usage: setup.sh [action]
#
# Options:
#   -a, --auto       Automatically detect action (install or update)
#   -i, --install    Install freqtrade
#   -u, --update     Update freqtrade
#   -h, --help       Display this help and exit
#
# Examples:
#   arguments.sh -f myfile.txt -vv arg1 arg2

set -eu

FREQTRADE_VERSION=stable
SETUP_FLAG=""
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.sh"

# print usage message
print_usage() {
  sed -Ene '/#\s?Usage: '"$BIN"'/,/^([^#]|$)/{p; /^([^#]|$)/q}' "$0" | sed -e '$d; s/#\s\?//'
}

log() {
  echo "===> $*"
}

auto() {
  if [[ -d "$FREQTRADE_DIR" ]]; then
    update
  else
    install
  fi
}

install() {
  if [[ -e "$FREQTRADE_DIR" ]]; then
    log "Already installed"
    return
  fi

  (
    log "Installing freqtrade version $FREQTRADE_VERSION"
    git clone --quiet --progress --depth 1 --branch "$FREQTRADE_VERSION" git@github.com:freqtrade/freqtrade.git "$FREQTRADE_DIR"
    cd "$FREQTRADE_DIR"
  )
  SETUP_FLAG="-i"
}

update() {
  (
    log "Updating freqtrade to version $FREQTRADE_VERSION"
    cd "$FREQTRADE_DIR"

    if ! [[ -z "$(git status --porcelain)" ]]; then
      log "Git repository contains uncommitted changes"
      git status --short
      read -rp "===> Do you want to reset git repository [y/N]? "
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "exiting"
        exit 1
      fi
      log "Resetting git repository"
      git reset --hard
    fi

    log "Fetching latest changes from remote"
    git fetch --quiet --prune origin "$FREQTRADE_VERSION"
    git checkout -B "$FREQTRADE_VERSION" "origin/$FREQTRADE_VERSION"
  )
  SETUP_FLAG="-u"
}

if [[ $# -gt 1 ]]; then
  echo "Too many arguments"
  exit 1
fi

case "${1-}" in
  ""|-a|--auto) auto;;
  -i|--install) install;;
  -u|--update)  update;;
  -h|--help)
    usage
    exit 0;;
  *)
    echo "Invalid argument: $1"
    exit 1;;
esac

(
  log "Running freqtrade setup script and install ploty + hyperopt dependencies"
  cd "$FREQTRADE_DIR"
  echo "N\ny\nyN" | ./setup.sh $SETUP_FLAG
)

log "Installing additional tools"
install_tools

log "Setting up backtest_results symlink"
# Replace freqtrade's backtest results with symlink to this repository backtest_results
test -d "$FREQTRADE_DIR/user_data/backtest_results" && rm -r "$FREQTRADE_DIR/user_data/backtest_results"
ln -fTsrv "$SCRIPT_DIR/../backtest_results" "$FREQTRADE_DIR/user_data/backtest_results"

log "done"
