#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/compose.yaml"
REQ_FILE="/workspace/envs/pytorch/requirements.user.txt"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
export HOST_USER="${HOST_USER:-$(id -un)}"

compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

usage() {
  cat <<'USAGE'
Usage: envs/pytorch/dev.sh [command]

Commands:
  build    Build image from nvcr.io/nvidia/pytorch:26.01-py3
  shell    Open an interactive shell (builds first)
  up       Start container in background (builds first)
  down     Stop and remove container (keeps named volumes)
  exec     Run command in running container (default: bash)
  freeze   Save user-site packages to envs/pytorch/requirements.user.txt
  install  Install envs/pytorch/requirements.user.txt into user site
USAGE
}

cmd="${1:-shell}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "${cmd}" in
  build)
    compose build --pull tamubo
    ;;
  shell)
    compose run --rm --build tamubo
    ;;
  up)
    compose up -d --build tamubo
    ;;
  down)
    compose down
    ;;
  exec)
    if [ "$#" -eq 0 ]; then
      set -- bash
    fi
    compose exec tamubo "$@"
    ;;
  freeze)
    compose run --rm tamubo bash -lc "python -m pip freeze --user | sort > ${REQ_FILE}"
    echo "Wrote ${REQ_FILE}"
    ;;
  install)
    compose run --rm tamubo bash -lc "if [ -f ${REQ_FILE} ]; then python -m pip install --user -r ${REQ_FILE}; else echo 'Missing ${REQ_FILE}'; exit 1; fi"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
