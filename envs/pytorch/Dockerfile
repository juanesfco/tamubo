# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3
FROM ${BASE_IMAGE}

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-lc"]

RUN set -eux; \
    if ! getent group "${USERNAME}" >/dev/null; then \
      groupadd --gid "${GID}" --non-unique "${USERNAME}"; \
    fi; \
    if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
      useradd --uid "${UID}" --gid "${GID}" --non-unique --create-home --shell /bin/bash "${USERNAME}"; \
    fi; \
    mkdir -p /workspace "/home/${USERNAME}/.local" "/home/${USERNAME}/.cache/pip"; \
    chown -R "${UID}:${GID}" /workspace "/home/${USERNAME}"

ENV HOME=/home/${USERNAME}
ENV PATH=${HOME}/.local/bin:${PATH}

WORKDIR /workspace
USER ${USERNAME}

CMD ["bash"]
