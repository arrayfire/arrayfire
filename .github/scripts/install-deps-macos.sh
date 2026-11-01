#!/usr/bin/env bash
# Installs the Homebrew packages for an ArrayFire CPU build on macOS. BLAS and
# LAPACK come from Apple's Accelerate framework; CMake 3.25 or newer must
# already be installed. It must run on macOS's bash 3.2.
set -euo pipefail

brew install boost ccache fftw fontconfig freeimage glfw ninja pkgconf
