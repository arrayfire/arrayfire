ArrayFire Binary Installation Instructions {#installing}
=====

Installing ArrayFire couldn't be easier. We ship installers for Windows,
OSX, and Linux. Although you could
[build ArrayFire from source](https://github.com/arrayfire/arrayfire), we
suggest using our pre-compiled binaries as they include the Intel Math
Kernel Library to accelerate linear algebra functions.

Please note that although our download page requires a valid login, registration
is free and downloading ArrayFire is also free. We request your contact
information so that we may notify you of software updates and occasionally
collect user feedback about our library.

In general, the installation process for ArrayFire looks like this:

1. Install prerequisites
2. [Download](http://arrayfire.com/download/) the ArrayFire installer for your
   operating system
3. Install ArrayFire
4. Test the installation
5. [Where to go for help?](#GettingHelp)

Below you will find instructions for:

* [Windows](#Windows)
* Linux
    * [Debian 8](#Debian)
    * [Ubuntu 14.04 and later](#Ubuntu)
    * [RedHat, Fedora, and CentOS](#RPM-distros)
* [Mac OSX (.sh and brew)](#OSX)

# <a name="Windows"></a> Windows

If you wish to use CUDA or OpenCL please ensure that you have also installed
support for these technologies from your video card vendor's website.

Next, [download](http://arrayfire.com/download/) and run the ArrayFire
installer. After installation, you'll need to add ArrayFire to the path for
all users:

1. Open Advanced System Settings:
    * Windows 8: Move the Mouse pointer to the bottom right corner of the
      screen, Right click, choose System. Then click "Advanced System Settings"
    * Windows 7: Open the Start Menu and Right Click on "Computer". Then choose
      Properties and click "Advanced System Settings"
2. In Advanced System Settings window, click on Advanced tab
3. Click on Environment Variables, then under System Variables, find PATH, and
   click on it.
4. In edit mode, append `%AF_PATH%/lib`. Make sure to separate `%AF_PATH%/lib`
   from any existing content using a semicolon (e.g.
   `EXISTING_PATHS;%AF_PATH%/lib;`). Other software may function incorrectly
   if this is not the case.

Finally, verify that the path addition worked correctly. You can do this by:

1. Open Visual Studio 2013. Open the `HelloWorld` solution which is located at
   `%AF_PATH%/examples/helloworld/helloworld.exe`.
2. Build and run the `helloworld` example. Use the "Solution Platform"
   drop-down to select from the CPU, CUDA, or OpenCL backends ArrayFire
   provides.

# Linux

## <a name="Debian"></a> Debian 8

First, install the prerequisite packages:

    # Install prerequisite packages:
    apt-get install libglfw3-dev cmake

    # Enable GPU support (OpenCL):
    apt-get install ocl-icd-libopencl1

If you wish to use CUDA,
[download](https://developer.nvidia.com/cuda-downloads) and install the latest
version.

Next, [download](http://arrayfire.com/download/) the ArrayFire installer for
your system. After you have the file, run the installer:

    ./arrayfire_*_Linux_x86_64.sh --exclude-subdir --prefix=/usr/local

## <a name="RPM-distros"></a> RedHat, Fedora, and CentOS

First, install the prerequisite packages:

    # Install prerequiste packages
    yum install glfw cmake

NOTE: On CentOS and Redhat, the `glfw` package is outdated and you will need
to compile it from source. Follow these
[instructions](https://github.com/arrayfire/arrayfire/wiki/GLFW-for-ArrayFire)
for more information on how to build and install GFLW.

If you wish to use CUDA,
[download](https://developer.nvidia.com/cuda-downloads) and install the latest
version.

Next, [download](http://arrayfire.com/download/) the ArrayFire installer for
your system. After you have the file, run the installer:

    ./arrayfire_*_Linux_x86_64.sh --exclude-subdir --prefix=/usr/local

## <a name="Ubuntu"></a> Ubuntu 14.04 and later

First, install the prerequisite packages:

### Ubuntu 16.04

    # Install prerequisite packages:
    sudo apt-get install libglfw3-dev cmake

### Ubuntu 14.04

    # Install prerequisite packages:
    sudo apt-get install cmake

Ubuntu 14.04 does not include the `libglfw3-dev` package in its
repositories. In order to install, you can either:

1. Build the library from source by following these
   [instructions](https://github.com/arrayfire/arrayfire/wiki/GLFW-for-ArrayFire),
   or
2. Install the library from a PPA as follows:

    sudo apt-add-repository ppa:keithw/glfw3
    sudo apt-get update
    sudo apt-get install glfw3

At this point, the installation should proceed identically for Ubuntu 14.04
and newer.

If your system has a CUDA GPU, we suggest downloading the latest drivers
from NVIDIA in the form of a Debian package and installing using the
package manager. At present, CUDA downloads can be found on the
[NVIDIA CUDA download page](https://developer.nvidia.com/cuda-downloads).
Follow NVIDIA's instructions for getting CUDA set up.

If you wish to use OpenCL, simply install the OpenCL ICD loader along
with any drivers required for your hardware.

    # Enable GPU support (OpenCL):
    apt-get install ocl-icd-libopencl1

### Special instructions for Tegra X1

**The ArrayFire binary installer for Tegra X1 requires at least JetPack 2.3 or
L4T 24.2 for Jetson TX1. This includes Ubuntu 16.04, CUDA 8.0 etc.**

You will also want to install the following packages when using ArrayFire on
the Tegra X1:

    sudo apt-get install libopenblas-dev liblapacke-dev

### Special instructions for Tegra K1

You will also want to install the following packages when using ArrayFire on
the Tegra K1:

    sudo apt-get install libatlas3gf-base libatlas-dev libfftw3-dev liblapacke-dev

Finally, [download](http://arrayfire.com/download/) ArrayFire for your
system. After you have the file, run the installer using:

    ./arrayfire_*_Linux_x86_64.sh --exclude-subdir --prefix=/usr/local

# <a name="OSX"></a> Mac OSX

On OSX there are several dependencies that are not integrated into the
operating system. The ArrayFire installer automatically satisfies these
dependencies using [Homebrew](http://brew.sh/).
If you don't have Homebrew installed on your system, the ArrayFire installer
will ask you do to so.

Simply [download](http://arrayfire.com/download) the ArrayFire installer
and double-click it to carry out the installation.

ArrayFire can also be installed through Homebrew directly using
`brew install arrayfire`; however, it will
not include MKL acceleration of linear algebra functions.

## Testing installation

Test ArrayFire after the installation process by building the example programs
as follows:

    cp -r /usr/local/share/ArrayFire/examples .
    cd examples
    mkdir build
    cd build
    cmake ..
    make

## <a name="GettingHelp"></a> Getting help

* Google Groups: https://groups.google.com/forum/#!forum/arrayfire-users
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting/)  |  [Support](http://arrayfire.com/support/)   |  [Training](http://arrayfire.com/training/)
* ArrayFire Blogs: http://arrayfire.com/blog/
* Email: <mailto:technical@arrayfire.com>
