ARG ISAACLAB_BASE_IMAGE_ARG

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# Use a pre-provisioned Isaac Lab 3.0 image; do not mutate its Python environment.
ENV PYTHONPATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/robot_lab
ENV PYTHONDONTWRITEBYTECODE=1

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace
