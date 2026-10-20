ARG ISAACLAB_BASE_IMAGE_ARG

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY . ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# Install task-discovery metadata in the base image's existing Isaac Lab environment.
RUN uv pip install --no-deps -e ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}
ENV PYTHONDONTWRITEBYTECODE=1

# Use the extension directory for project-relative scripts and task discovery.
WORKDIR ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}
