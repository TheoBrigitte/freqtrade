#!/usr/bin/env bash

FREQTRADE_DIR="$SCRIPT_DIR/../freqtrade"

JQ_VERSION="1.7.1"
JQ_BIN="${SCRIPT_DIR}/jq"

FZF_VERSION="0.56.3"
FZF_BIN="${SCRIPT_DIR}/fzf"

get_args() {
  args=()

  args+=(--config "$SCRIPT_DIR/../config/config_base.json")
  #args+=(--config "$SCRIPT_DIR/../config/config_baseline.json")

  if [ "$FREQTRADE_MODE" == "futures" ]; then
    args+=(--config "$SCRIPT_DIR/../config/config_futures.json")

    args+=(--config "$SCRIPT_DIR/../pairlist/pairlist_futures.json")
    #args+=(--config "$SCRIPT_DIR/../pairlist/monthly_60_USDT_0,0_minprice_current.json")
  else
    args+=(--config "$SCRIPT_DIR/../pairlist/pairlist_spot.json")
    #args+=(--config "$SCRIPT_DIR/../pairlist/monthly_60_USDT_0,05_minprice_current_1.json")

    #args+=(--config "$SCRIPT_DIR/../config/config_spot_usdc.json")
    #args+=(--config "$SCRIPT_DIR/../pairlist/hyperliquid-usdc-static_spot.json")
  fi

  args+=(--config "$SCRIPT_DIR/../pairlist/blacklist.json")
  #args+=(--config "$SCRIPT_DIR/../config/config_fix10100.json")
  args+=(--config "$SCRIPT_DIR/../config/config_stake.json")

  args+=(--user-data-dir $FREQTRADE_DIR/user_data)

  echo "${args[@]}"
}

install_tool() {
    local binary="$(basename $1)"; shift
    local version="$1"; shift
    local tmp_path="$1"; shift
    local url="$1" ; shift
    local destination="${SCRIPT_DIR}/${binary}"

    if [[ ! -f "${destination}" ]]; then
        echo "> Installing ${binary}"
        tmpdir="$(mktemp -dt freqtrade-install.XXXXXX)"
        blob="$(basename "${url}")"

        curl -fsL -o "${tmpdir}/${blob}" "${url}"

        case "${blob}" in
          *.tar*)
            tar -C "${tmpdir}" -xf "${tmpdir}/${blob}"
            mv "${tmpdir}/${tmp_path}" "${destination}" ;;
          *.zip)
            unzip -d "${tmpdir}" -qo "${tmpdir}/${blob}"
            mv "${tmpdir}/${tmp_path}" "${destination}" ;;
          *)
            mv "${tmpdir}/${blob}" "${destination}" ;;
        esac

        rm -rf "$tmpdir"
        chmod +x "${destination}"
    fi
}

install_tools() {
  os=""
  case "$OSTYPE" in
    linux*)   os="linux" ;;
    *)        echo "unsupported OS: $OSTYPE"; exit ;;
  esac

  arch=""
  case "$(uname -m)" in
    x86_64) arch="amd64" ;;
    arm*)   arch="arm64" ;;
    aarch*) arch="arm64" ;;
    *)     echo "unsupported architecture: $(uname -m)"; exit ;;
  esac

  install_tool "${JQ_BIN}" "${JQ_VERSION}" "" "https://github.com/stedolan/jq/releases/download/jq-${JQ_VERSION}/jq-${os}-${arch}"
  install_tool "${FZF_BIN}" "${FZF_VERSION}" "fzf" "https://github.com/junegunn/fzf/releases/download/v${FZF_VERSION}/fzf-${FZF_VERSION}-${os}_${arch}.tar.gz"
}
