# NCCL Build & Extract Utility

This script (`build-extract-lib.sh`) automates building NCCL from a specified repository and branch inside an NGC PyTorch container, then extracts the resulting NCCL libraries to your host filesystem. It is useful for quickly building custom NCCL versions for testing or deployment.

## Docker Installation (If Needed)

If you do not have Docker and the NVIDIA Container Toolkit installed, run the following commands:

```bash
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

distribution=$(. /os-release;echo $ID$VERSION_ID) \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
```

## Usage

```bash
./build-extract-lib.sh <ngc_tag> <branch_or_ref> [nccl_repo_url] [deployment]
```

- `<ngc_tag>`: The NGC PyTorch container tag (e.g., `23.12`, `24.01`, etc).
- `<branch_or_ref>`: The git branch, tag, or commit to build from (e.g., `main`, `v2.20.3-1`, etc).
- `[nccl_repo_url]` (optional): The NCCL git repository URL. Defaults to `https://gitlab-master.nvidia.com/ai-efficiency/nccl.git` if not provided.
- `[deployment]` (optional): Deployment string passed to the build process (used by `gen-package-data.sh`).

## Example

Build NCCL from the `main` branch using the `24.01` NGC PyTorch container:

```bash
./build-extract-lib.sh 24.01 main
```

Build NCCL from a custom repository and branch:

```bash
./build-extract-lib.sh 24.08 v2.27 https://gitlab-master.nvidia.com/sirshakd/nccl-inspector
```

## What the Script Does

1. Launches a temporary NGC PyTorch container for the specified tag.
2. Clones the specified NCCL repository and branch/ref inside the container.
3. Builds NCCL with MPI and CUDA support.
4. Extracts the built NCCL libraries and package metadata to a local directory named `container-libs/ngc-pytorch-<ngc_tag>-<git_info>`.
5. Cleans up all temporary files and containers.

## Requirements

- Docker
- jq
- A compatible NGC PyTorch container tag (see [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch))
- Internet access to clone the NCCL repository

## Output

After running, you will find the built NCCL libraries and a `package.json` with version and commit info in a directory like:

```
container-libs/ngc-pytorch-24.01-v2.20.3-1-abcdefg
```

## Notes

- The script will overwrite any existing output directory with the same name.
- Adjust the CUDA architecture and other build flags in the script as needed for your hardware.
- The script expects `gen-package-data.sh` to be present in `etc/` relative to the script location.

---

For troubleshooting or advanced usage, see comments in the script itself.
