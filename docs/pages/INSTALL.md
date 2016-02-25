ArrayFire binary installation instructions {#installing}
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

Below you will find instructions for

* [Windows](#Windows)
* Linux including
    * [Debian 8](#Debian)
    * [Ubuntu 14.04 and later](#Ubuntu)
    * [RedHat, Fedora, and CentOS](#RPM-distros)
* [Mac OSX (.sh and brew)](#OSX)

# <a name="Windows"></a> Windows

If you wish to use CUDA or OpenCL please ensure that you have also installed
support for these technologies from your video card vendor's website.

Next [download](http://arrayfire.com/download/) and run the ArrayFire installer.
After it has completed, you need to add ArrayFire to the path for all users.

1. Open Advanced System Settings:
    * Windows 8: Move the Mouse pointer to the bottom right corner of the
      screen, Right click, choose System. Then click "Advanced System Settings"
    * Windows 7: Open the Start Menu and Right Click on "Computer". Then choose
      Properties and click "Advanced System Settings"
2. In Advanced System Settings window, click on Advanced tab
3. Click on Environment Variables, then under System Variables, find PATH, and
   click on it.
4. In edit mode, append %AF_PATH%/lib. NOTE: Ensure that there is a semi-colon
   separating %AF_PATH%/lib from any existing content (e.g.
   EXISTING_PATHS;%AF_PATH%/lib;) otherwise other software may not function
   correctly.

Finally, verify that the path addition worked correctly. You can do this by:

1. Open Visual Studio 2013. Open the HelloWorld solution which is located at
   `%AF_PATH%/examples/helloworld/helloworld.exe`.
2. Build and run the helloworld example. Be sure to, select the
   platform/configuration of your choice using the platform drop-down (the
   options are CPU, CUDA, and OpenCL) and Solution Configuration drop down
   (options of Release and Debug) menus. Run the helloworld example

# Linux

## <a name="Debian"></a> Debian 8

First install the prerequisite packages:

    # Prerequisite packages:
    apt-get install libfreeimage-dev libatlas3gf-base libfftw3-dev libglew-dev libglewmx-dev libglfw3-dev cmake

    # Enable GPU support (OpenCL):
    apt-get install ocl-icd-libopencl1

If you wish to use CUDA, please
[download the latest version of CUDA](https://developer.nvidia.com/cuda-zone)
and install it on your system.

Next [download](http://arrayfire.com/download/) ArrayFire. After you have the
file, run the installer.

    ./arrayfire_*_Linux_x86_64.sh --exclude-subdir --prefix=/usr/local

## <a name="RPM-distros"></a> RedHat, Fedora, and CentOS

First install the prerequisite packages:

    # Install prerequiste packages
    yum install freeimage atlas fftw libGLEW libGLEWmx glfw cmake

On Centos and Redhat the `glfw` package is outdated and you will need to compile
it from source. Please
[these instructions](https://github.com/arrayfire/arrayfire/wiki/GLFW-for-ArrayFire).

If you wish to use CUDA, please
[download the latest version of CUDA](https://developer.nvidia.com/cuda-downloads)
and install it on your system.

Next [download](http://arrayfire.com/download/) ArrayFire. After you have the
file, run the installer.

    ./arrayfire_*_Linux_x86_64.sh --exclude-subdir --prefix=/usr/local

## <a name="Ubuntu"></a> Ubuntu 14.04 and later

First install the prerequisite packages:

    # Prerequisite packages:
    sudo apt-get install libfreeimage-dev libatlas3gf-base libfftw3-dev cmake

Ubuntu 14.04 will not have the libglfw3-dev package in its repositories. You can either build the
library from source (following the
[instructions listed here](https://github.com/arrayfire/arrayfire/wiki/GLFW-for-ArrayFire)) or
install the library from a PPA as follows:

    sudo apt-add-repository ppa:keithw/glfw3
    sudo apt-get update
    sudo apt-get install glfw3

After this point, the installation should proceed identically to Ubuntu 14.10 or newer.

If your system has a CUDA GPU, we suggest downloading the latest drivers
from NVIDIA in the form of a Debian package and installing using the
package manager. At present, CUDA downloads can be found on the
[NVIDIA CUDA download page](https://developer.nvidia.com/cuda-downloads)
Follow NVIDIA's instructions for getting CUDA set up.

If you wish to use OpenCL, simply install the OpenCL ICD loader along
with any drivers required for your hardware.

    # Enable GPU support (OpenCL):
    apt-get install ocl-icd-libopencl1

### Special instructions for Tegra K1
If you are using ArrayFire on the Tegra K1 also install these packages:

    sudo apt-get install libatlas3gf-base libatlas-dev libfftw3-dev liblapacke-dev

In addition to these packages, you will need to compile GLFW3 from source
using the instructions above.

Finally, [download](http://arrayfire.com/download/) ArrayFire. After you have
the file, run the installer using:

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

After ArrayFire is installed, you can build the example programs as follows:

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
