---
name: Bug Report
about: Create a bug report to help us improve ArrayFire
title: "[BUG]"
labels: 'bug'
assignees: ''
---

<!-- One to two sentences discription of the bug -->

Description
===========
<!--
* Additional details regarding the bug
* Did you build ArrayFire yourself or did you use the official installers
* Which backend is experiencing this issue? (CPU, CUDA, OpenCL)
* Do you have a workaround?
* Can the bug be reproduced reliably on your system?
* A clear and concise description of what you expected to happen.
* Run your executable with AF_TRACE=all and AF_PRINT_ERRORS=1 environment
  variables set.
* Screenshot or terminal output of the results
-->

Reproducible Code and/or Steps
------------------------------
<!--
* Steps or code snippet that can reproduce the bug
* A full example will allow us to debug and fix the bug faster
-->

System Information
------------------
<!--
Please provide the following information:
1. ArrayFire version
2. Devices installed on the system
3. (optional) Output from the af::info() function if applicable.
4. Output from the following scripts:

Run one of the following commands based on your OS

Linux:
```sh
lsb_release -a
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu="name,memory.total,driver_version" --format=csv -i 0
else
  echo "nvidia-smi not found"
fi
if command -v /opt/rocm/bin/rocm-smi >/dev/null; then
  /opt/rocm/bin/rocm-smi --showproductname
else
  echo "rocm-smi not found."
fi
if command -v clinfo > /dev/null; then
  clinfo
else
  echo "clinfo not found."
fi
```

Windows:
Download clinfo from https://github.com/Oblomov/clinfo

If you have NVIDIA GPUs. Run nvidia-smi usually located in
C:\Program Files\NVIDIA Corporation\NVSMI

Provide driver version for your GPU. (This is vendor specific)
-->

Checklist
---------

- [ ] Using the latest available ArrayFire release
- [ ] GPU drivers are up to date
