FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/tamubo-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    openmpi-bin \
    libopenmpi-dev \
    python3 \
    python3-pip \
    python3-venv \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade pip \
    && python -m pip install \
    numpy \
    scipy \
    scikit-learn \
    pybind11 \
    pytest

RUN set -eux; \
    if ! getent group "${USERNAME}" >/dev/null; then \
      groupadd --gid "${GID}" --non-unique "${USERNAME}"; \
    fi; \
    if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
      useradd --uid "${UID}" --gid "${GID}" --non-unique --create-home --shell /bin/bash "${USERNAME}"; \
    fi; \
    chown -R "${UID}:${GID}" "${VIRTUAL_ENV}" "/home/${USERNAME}"

ENV HOME=/home/${USERNAME}
USER ${USERNAME}

CMD ["/bin/bash"]
