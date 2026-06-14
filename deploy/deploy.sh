#!/usr/bin/env bash
###############################################################################
# deploy/deploy.sh
#
# Part 3 / Part 7 deliverable.
#
# Builds the Docker image, pushes it to Docker Hub, then SSHes into the
# configured EC2 instance to pull and run the new image, replacing any
# previously running container.
#
# Required environment variables (export before running, or pass via CI secrets):
#   DOCKER_USERNAME   - Docker Hub username
#   DOCKER_PASSWORD   - Docker Hub access token / password
#   DOCKER_IMAGE_NAME - e.g. yourdockerhubuser/mlops-inference
#   EC2_HOST          - EC2 public IP or DNS
#   EC2_USER          - SSH user (e.g. ubuntu)
#   EC2_SSH_KEY_PATH  - path to the PEM private key file (CI writes this from secret)
#   IMAGE_TAG         - tag to deploy (default: latest)
#
# Usage:
#   ./deploy/deploy.sh build-and-push     # build image + push to Docker Hub
#   ./deploy/deploy.sh remote-deploy      # SSH to EC2, pull + run image
#   ./deploy/deploy.sh all                # do both
###############################################################################

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${DOCKER_IMAGE_NAME}:${IMAGE_TAG}"

build_and_push() {
    echo ">>> Building Docker image: ${FULL_IMAGE}"
    docker build -t "${FULL_IMAGE}" -t "${DOCKER_IMAGE_NAME}:latest" .

    echo ">>> Logging in to Docker Hub"
    echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin

    echo ">>> Pushing ${FULL_IMAGE}"
    docker push "${FULL_IMAGE}"
    docker push "${DOCKER_IMAGE_NAME}:latest"
}

remote_deploy() {
    echo ">>> Deploying ${DOCKER_IMAGE_NAME}:latest to ${EC2_USER}@${EC2_HOST}"

    ssh -o StrictHostKeyChecking=no -i "${EC2_SSH_KEY_PATH}" "${EC2_USER}@${EC2_HOST}" bash -s <<EOF
        set -e
        echo ">>> Pulling latest image"
        docker pull ${DOCKER_IMAGE_NAME}:latest

        echo ">>> Stopping old container (if running)"
        docker stop mlops-inference || true
        docker rm mlops-inference || true

        echo ">>> Starting new container"
        docker run -d \
            --name mlops-inference \
            --restart unless-stopped \
            -p 80:8000 \
            -p 8000:8000 \
            -e SLACK_WEBHOOK_URL="\${SLACK_WEBHOOK_URL:-}" \
            ${DOCKER_IMAGE_NAME}:latest

        echo ">>> Waiting for service to become healthy"
        sleep 5
        curl -f http://localhost:8000/health
EOF

    echo ">>> Deployment complete."
}

case "${1:-all}" in
    build-and-push)
        build_and_push
        ;;
    remote-deploy)
        remote_deploy
        ;;
    all)
        build_and_push
        remote_deploy
        ;;
    *)
        echo "Usage: $0 {build-and-push|remote-deploy|all}"
        exit 1
        ;;
esac
