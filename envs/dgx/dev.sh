#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

IMAGE="${IMAGE:-tamubo-dgx:cuda13-arm64}"
CONTAINER_NAME="${CONTAINER_NAME:-tamubo-dgx-dev}"
PLATFORM="${PLATFORM:-linux/arm64}"

HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"

docker_run_flags=(
  --platform "${PLATFORM}"
  --gpus all
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "${PROJECT_ROOT}:/workspace"
  -w /workspace
)

build_image() {
  docker build \
    --platform "${PLATFORM}" \
    --build-arg "USERNAME=${HOST_USER}" \
    --build-arg "UID=${HOST_UID}" \
    --build-arg "GID=${HOST_GID}" \
    -t "${IMAGE}" \
    -f "${DOCKERFILE}" \
    "${PROJECT_ROOT}"
}

usage() {
  cat <<USAGE
Usage: envs/dgx/dev.sh [command]

Commands:
  build      Build ${IMAGE}
  shell      Build and open an interactive shell
  up         Build and start a named background container
  exec       Run a command in the background container (default: bash)
  down       Stop and remove the background container
  verify     Run basic CUDA/MPI/Python checks
  help       Show this help

Environment overrides:
  IMAGE=${IMAGE}
  CONTAINER_NAME=${CONTAINER_NAME}
  PLATFORM=${PLATFORM}
  HOST_USER=${HOST_USER}
  HOST_UID=${HOST_UID}
  HOST_GID=${HOST_GID}
USAGE
}

cmd="${1:-shell}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "${cmd}" in
  build)
    build_image
    ;;
  shell)
    build_image
    docker run --rm -it "${docker_run_flags[@]}" "${IMAGE}" bash
    ;;
  up)
    build_image
    if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
      echo "Container ${CONTAINER_NAME} already exists. Use '${SCRIPT_DIR}/dev.sh down' first."
      exit 1
    fi
    docker run -d \
      --name "${CONTAINER_NAME}" \
      "${docker_run_flags[@]}" \
      "${IMAGE}" \
      bash -lc "sleep infinity"
    ;;
  profile)
    build_image
    docker run --rm -it \
      "${docker_run_flags[@]}" \
      --cap-add=SYS_ADMIN \
      --user root \
      "${IMAGE}" \
      bash
    ;;
  exec)
    if [ "$#" -eq 0 ]; then
      set -- bash
    fi
    docker exec -it "${CONTAINER_NAME}" "$@"
    ;;
  down)
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    ;;
  verify)
    docker run --rm "${docker_run_flags[@]}" "${IMAGE}" bash -lc \
      'whoami && id && uname -m && which python && python --version && nvidia-smi && nvcc --version && mpirun --version | sed -n "1p" && nsys --version'
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
