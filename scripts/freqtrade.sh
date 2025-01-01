#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/common.sh"

action=$1
shift

args=($(get_args))

set -x
$FREQTRADE_DIR/.venv/bin/freqtrade $action "${args[@]}" $@
