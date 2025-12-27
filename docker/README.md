# Docker Setup

This folder contains Docker configuration for running the project in containerized environments.

## Files

- `Dockerfile.cpu` - CPU-only Docker image
- `Dockerfile.gpu` - GPU-enabled Docker image (requires nvidia-docker v2.0)
- `launch-docker-cpu.sh` - Launch script for CPU container
- `launch-docker-gpu.sh` - Launch script for GPU container

## Build Images

```bash
# From project root:

# CPU image
docker build -f docker/Dockerfile.cpu -t gdl-image-cpu .

# GPU image (requires nvidia-docker)
docker build -f docker/Dockerfile.gpu -t gdl-image .
```

## Run Containers

```bash
# CPU (from project root)
./docker/launch-docker-cpu.sh $(pwd)

# GPU (from project root)
./docker/launch-docker-gpu.sh $(pwd)
```

## Requirements

- Docker installed
- For GPU: nvidia-docker v2.0 ([installation guide](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)))
- Project must have `requirements.txt` at root for build
