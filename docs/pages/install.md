ArrayFire Binary Installation Instructions {#installing}
=====

Installing ArrayFire couldn't be easier. We ship installers for Windows,
OSX, and Linux. Although you could
[build ArrayFire from source](https://github.com/arrayfire/arrayfire), we
suggest using our pre-compiled binaries as they include the Intel Math
Kernel Library to accelerate linear algebra functions.

You can download official ArrayFire installers by navigating to
http://arrayfire.com/download/ . Select the installer for your architecture and
operating system.

Make sure you have the latest drivers installed on your system before using
ArrayFire. If you are going to be targeting the CPU using OpenCL, you will need
to have the OpenCL **runtime*** installed on your system. You can download the
drivers and runtimes from the device vendors website.

# Install Instructions

* [Windows](#Windows)
* Linux
    * [Debian derivatives (Ubuntu etc.)](#Debian)
    * [RedHat, Fedora, and CentOS](#RPM-distros)
* [Mac OSX (.sh and brew)](#OSX)

## <a name="Windows"></a> Windows

Additionally you will need to install the 64-bit(x64) visual studio 2015 runtime
 libraries from
 [here](https://www.microsoft.com/en-in/download/details.aspx?id=48145).

If you chose not to modify PATH during the installation procedure, you'll need
to manually add ArrayFire to the path for all users. Simply append
`%AF_PATH%/lib `to the PATH variable so that the loader can find ArrayFire DLLs.

## Linux

Run the ArrayFire installer for your system. The prefix command determines the
location of the installed files. You can install ArrayFire in any directory but
it is recommended you install in /opt.

    ./Arrayfire_*_Linux_x86_64.sh --include-subdir --prefix=/opt
    echo /opt/arrayfire/lib > /etc/ld.so.conf.d/arrayfire.conf
    ldconfig

In addition to the drivers and runtime files, you will also need to install
FreeImage and some additional dependencies if you are installing the graphics
version.

### <a name="Debian"></a> Debian 8 & Debian derivatives such as Ubuntu(14.04 or later)

    apt update
    apt install build-essential libfreeimage3

    # Additional dependencies for graphics installers
    apt install libfontconfig1 libglu1-mesa

### <a name="RPM-distros"></a> RedHat, Fedora, and CentOS

    # Install prerequiste packages
    yum install freeimage

    # Additional dependencies for graphics installers
    yum install fontconfig mesa-libGLU

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

## <a name="OSX"></a> Mac OSX

Simply [download](http://arrayfire.com/download/) ArrayFire for your
system. After you have the file, run the installer from command line using:

    sudo installer -pkg Arrayfire-*_OSX.pkg -target /

## Testing installation

Test ArrayFire on Unix style platforms after the installation process by
building the example programs as follows:

    cp -r /opt/arrayfire/share/ArrayFire/examples /tmp/examples
    cd /tmp/examples
    mkdir build
    cd build
    cmake -DASSETS_DIR:PATH=/tmp .. 
    make

On Windows, open the CMakeLists.txt file from CMake-GUI and set ASSETS\_DIR
varible to the parent folder of examples folder. Once the project is configured
and generated, you can build and run the examples from Visual Studio.

## <a name="GettingHelp"></a> Getting help

* Google Groups: https://groups.google.com/forum/#!forum/arrayfire-users
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting/)  |  [Support](http://arrayfire.com/support/)   |  [Training](http://arrayfire.com/training/)
* ArrayFire Blogs: http://arrayfire.com/blog/
* Email: <mailto:technical@arrayfire.com>
