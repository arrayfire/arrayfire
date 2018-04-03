ArrayFire Binary Installation Instructions {#installing}
=====

Installing ArrayFire couldn't be easier. We ship installers for Windows,
OSX, and Linux. Although you could
[build ArrayFire from source](https://github.com/arrayfire/arrayfire), we
suggest using our pre-compiled binaries as they include the Intel Math
Kernel Library to accelerate linear algebra functions.

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

If you wish to use CUDA or OpenCL, please ensure that you have installed
the drivers for the video card(s) you intend to use from the respective
vendor's website. Please also install the 64-bit(x64) visual studio 2015
runtime libraries from
 [here](https://www.microsoft.com/en-in/download/details.aspx?id=48145).

Next, [download](http://arrayfire.com/download/) and run the ArrayFire
installer. If you chose not to modify PATH during the installation
procedure, you'll need to manually add ArrayFire to the path for all
users. You can follow the below steps to modify PATH environment variable:

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

1. Open Visual Studio. Open the `HelloWorld` solution which is located at
   `%AF_PATH%/examples/helloworld/helloworld.exe`.
2. Build and run the `helloworld` example. Use the "Solution Platform"
   drop-down to select from the CPU, CUDA, or OpenCL backends ArrayFire
   provides.

# Linux

If you wish to use CUDA or OpenCL, please ensure that you have installed
the drivers for the video card(s) you intend to use from the vendor's website.

First, install the prerequisite packages:

### <a name="Debian"></a> Debian 8

    apt update
    apt install build-essential libfreeimage3
    echo /opt/arrayfire/lib > /etc/ld.so.conf.d/arrayfire.conf
    ldconfig

    # Additional dependencies for graphics installers
    apt install libfontconfig1 libglu1-mesa

### <a name="RPM-distros"></a> RedHat, Fedora, and CentOS

    # Install prerequiste packages
    yum install freeimage
    echo /opt/arrayfire/lib > /etc/ld.so.conf.d/arrayfire.conf
    ldconfig

    # Additional dependencies for graphics installers
    yum install fontconfig mesa-libGLU

### Ubuntu 16.04, 14.04

    # Install prerequisite packages:
    sudo apt-get install cmake

    # Enable GPU support (OpenCL):
    apt-get install ocl-icd-libopencl1

Next, [download](http://arrayfire.com/download/) the ArrayFire installer for
your system. After you have the file, run the installer:

    ./Arrayfire_*_Linux_x86_64.sh --include-subdir --prefix=/opt

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

    ./Arrayfire_*_Linux_x86_64.sh --include-subdir --prefix=/opt

# <a name="OSX"></a> Mac OSX

Simply [download](http://arrayfire.com/download/) ArrayFire for your
system. After you have the file, run the installer from command line using:

    sudo installer -pkg Arrayfire-*_OSX.pkg -target /

The ArrayFire installer automatically installs most of the dependencies
which include MKL acceleration for linear algebra functions.

## Testing installation

Test ArrayFire on Unix style platforms after the installation process by
building the example programs as follows:

    cp -r /opt/arrayfire/share/ArrayFire/examples .
    cd examples
    mkdir build
    cd build
    cmake .. -DASSETS_DIR:PATH=/opt/arrayfire/share/ArrayFire
    make

On Windows, open the CMakeLists.txt file from CMake-GUI and set ASSETS\_DIR
varible to the parent folder of examples folder. Once the project is configured
and generated, you can build and run the examples from Visual Studio.

## <a name="GettingHelp"></a> Getting help

* Google Groups: https://groups.google.com/forum/#!forum/arrayfire-users
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting/)  |  [Support](http://arrayfire.com/support/)   |  [Training](http://arrayfire.com/training/)
* ArrayFire Blogs: http://arrayfire.com/blog/
* Email: <mailto:technical@arrayfire.com>
