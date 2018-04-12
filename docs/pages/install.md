# ArrayFire Installer {#installing}

Installing ArrayFire couldn't be easier. Navigate to
https://arrayfire.com/download and download the installer for your architecture
and operating system. Although you could [build ArrayFire from
source](https://github.com/arrayfire/arrayfire), we recommend using our
installers as we have packaged together all of the necessary dependencies to
give you the best performance.

We provide installers for Windows, Linux, and macOS. There are two installers
for each operating system: one with graphics support and the other without
graphics support. Download the installer with graphics support if you would like
to be able to do high performance visualizations using our
[Forge](https://github.com/arrayfire/forge) library. Otherwise, download the
installer without graphics support.

Make sure you have the latest device drivers installed on your system before
using ArrayFire. If you are going to be targeting the CPU using ArrayFire’s
OpenCL backend, you will need to have the OpenCL **runtime** installed on your
system. Drivers and runtimes should be downloaded and installed from your device
vendor’s website.

# Install Instructions

* [Windows](#Windows)
* [Linux](#Linux)
* [macOS](#macOS)

## <a name="Windows"></a> Windows

Prior to installing ArrayFire on Windows,
[download](https://www.microsoft.com/en-in/download/details.aspx?id=48145)
install the Visual Studio 2015 (x64) runtime libraries.

Once you have downloaded the ArrayFire installer, execute the installer as you
normally would on Windows. If you choose not to modify the path during the
installation procedure, you'll need to manually add ArrayFire to the path for
all users. Simply append `%AF_PATH%/lib` to the PATH variable so that the loader
can find ArrayFire DLLs.

For more information on using ArrayFire on Windows, visit the following
[page](http://arrayfire.org/docs/using_on_windows.htm).

## <a name="Linux"></a> Linux

Once you have downloaded the ArrayFire installer, execute the installer from the
terminal as shown below. Set the `--prefix` argument to the directory you would
like to install ArrayFire to - we recommend `/opt`.

    ./Arrayfire_*_Linux_x86_64.sh --include-subdir --prefix=/opt

Given sudo permissions, you can add the ArrayFire libraries via `ldconfig` like
so:

    echo /opt/arrayfire/lib > /etc/ld.so.conf.d/arrayfire.conf
    sudo ldconfig

Otherwise, you will need to set the `LD_LIBRARY_PATH` environment variable in
order to let your shared library loader find the ArrayFire libraries.

For more information on using ArrayFire on Linux, visit the following
[page](http://arrayfire.org/docs/using_on_linux.htm).

### Graphics support

ArrayFire allows you to do high performance visualizations via our
[Forge](https://github.com/arrayfire/forge) library. On Linux, there are a few
dependencies you will need to install to enable graphics support:

FreeImage
Fontconfig
GLU (OpenGL Utility Library)

We show how to install these dependencies on common Linux distributions:

__Debian, Ubuntu (14.04 and above), and other Debian derivatives__

    apt install build-essential libfreeimage3 libfontconfig1 libglu1-mesa

__Fedora, Redhat, CentOS__

    yum install freeimage fontconfig mesa-libGLU


## <a name="macOS"></a> macOS

Once you have downloaded the ArrayFire installer, execute the installer by
either double clicking on the ArrayFire `pkg` file or running the following
command from your terminal:

    sudo installer -pkg Arrayfire-*_OSX.pkg -target /

For more information on using ArrayFire on macOS, visit the following
[page](http://arrayfire.org/docs/using_on_osx.htm).

## NVIDIA Tegra devices

ArrayFire is capable of running on TX1 and TX2 devices. The TK1 is no longer
supported.

Prior to installing ArrayFire, make sure you have the latest version of JetPack
(v2.3 and above) or L4T (v24.2 and above) on your device.

### Tegra prerequisites

The following dependencies are required for Tegra devices:

    sudo apt install libopenblas-dev liblapacke-dev

## Testing installation

After ArrayFire is finished installing, we recommend building and running a few
of the provided examples to verify things are working as expected.

On Unix-like systems:

    cp -r /opt/arrayfire/share/ArrayFire/examples /tmp/examples
    cd /tmp/examples
    mkdir build
    cd build
    cmake -DASSETS_DIR:PATH=/tmp ..
    make
    ./helloworld/helloworld_{cpu,cuda,opencl}

On Windows, open the CMakeLists.txt file from CMake-GUI and set `ASSETS_DIR`
variable to the parent folder of examples folder. Once the project is configured
and generated, you can build and run the examples from Visual Studio.

## <a name="GettingHelp"></a> Getting help

* Google Groups: https://groups.google.com/forum/#!forum/arrayfire-users
* ArrayFire Services:  [Consulting](https://arrayfire.com/consulting/)  |  [Support](https://arrayfire.com/support/)   |  [Training](https://arrayfire.com/training/)
* ArrayFire Blogs: http://arrayfire.com/blog/
* Email: <mailto:technical@arrayfire.com>
