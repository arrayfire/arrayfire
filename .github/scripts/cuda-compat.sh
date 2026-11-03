#!/usr/bin/env bash
# Decides whether the CUDA tests can run on the host's driver as it is, or need
# NVIDIA's forward-compatibility package: reads nvidia-smi's output on stdin and
# prints "native <version>" or "compat <version>", with the CUDA version the
# driver supports. A driver supports the toolkits up to its own CUDA version;
# ArrayFire's JIT loads PTX made by NVRTC, which minor-version compatibility
# doesn't cover, so an older driver needs cuda-compat-<major>-<minor>.
#
# Usage: nvidia-smi | cuda-compat.sh <toolkit major.minor>
set -euo pipefail

[[ $# -eq 1 && $1 =~ ^([0-9]+)\.([0-9]+)$ ]] ||
  { echo "usage: nvidia-smi | $0 <toolkit major.minor>" >&2; exit 2; }
toolkit_major=${BASH_REMATCH[1]} toolkit_minor=${BASH_REMATCH[2]}

# "CUDA Version: 12.8" before driver 600, "CUDA UMD Version: 13.4" from then on.
driver=$(sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\2/p' | head -n 1)
[[ -n $driver ]] || { echo "nvidia-smi didn't say which CUDA version the driver supports" >&2; exit 1; }
driver_major=${driver%.*} driver_minor=${driver#*.}

if ((driver_major > toolkit_major ||
     (driver_major == toolkit_major && driver_minor >= toolkit_minor))); then
  echo "native $driver"
else
  echo "compat $driver"
fi
