# ArrayFire Installer {#installing}

Installing ArrayFire couldn't be easier. Navigate to
https://arrayfire.com/download and download the appropriate installer for the
target architecture and operating system. Although ArrayFire can be [built
from source](https://github.com/arrayfire/arrayfire), the installers
conveniently package necessary dependencies.

Install the latest device drivers before using ArrayFire. If you target the
CPU using ArrayFire’s OpenCL backend, install the OpenCL runtime. Drivers and
runtimes should be downloaded and installed from each device vendor's website.

# Install Instructions {#InstallInstructions}

* [Windows](#Windows)
* [Linux](#Linux)
* [macOS](#macOS)

## Windows {#Windows}

Once the ArrayFire has been downloaded, run the installer.

The installer offers the option to automatically add ArrayFire to the path for
all users. If the installer did not do this, simply append `%%AF_PATH%/lib` to
the PATH variable so that the loader can find ArrayFire DLLs.

For more information on using ArrayFire on Windows, visit the following
[page](http://arrayfire.org/docs/using_on_windows.htm).

## Linux {#Linux}

There are two ways to install ArrayFire on Linux.
1. Package Manager
2. Using the ArrayFire Linux Installer

As of today, approach (1) is only supported for Ubuntu 18.04 and 20.04. Please
go through [the GitHub
wiki[page](https://github.com/arrayfire/arrayfire/wiki/Install-ArrayFire-From-Linux-Package-Managers)
for detailed instructions.

For approach (2), once the ArrayFire installer is downloaded, execute the
installer from the terminal as shown below. Set the `--prefix` argument to the
target install directory; we recommend `/opt`.

    ./ArrayFire_*_Linux_x86_64.sh --include-subdir --prefix=/opt

Given sudo permissions, the ArrayFire libraries can be added to the path via
`ldconfig` like so:

    echo /opt/arrayfire/lib64 > /etc/ld.so.conf.d/arrayfire.conf
    sudo ldconfig

Otherwise, the `LD_LIBRARY_PATH` environment variable can be set so that the
shared library loader can find the ArrayFire libraries.

For more information on using ArrayFire on Linux, visit the following
[page](http://arrayfire.org/docs/using_on_linux.htm).

### Graphics support

ArrayFire enables high-performance visualizations via the
[Forge](https://github.com/arrayfire/forge) library. On Linux, there are a few
dependencies to install to enable graphics support:

* FreeImage
* Fontconfig
* GLU (OpenGL Utility Library)

To install these dependencies on common Linux distributions:

__Debian, Ubuntu (14.04 and above), and other Debian derivatives__

    apt install build-essential libfreeimage3 libfontconfig1 libglu1-mesa

__Fedora, Redhat, CentOS__

    yum install freeimage fontconfig mesa-libGLU


## macOS {#macOS}

Once the ArrayFire installer has been downloaded, execute the installer by
either double-clicking on the ArrayFire `pkg` file or running the following
command:

    sudo installer -pkg Arrayfire-*_OSX.pkg -target /

For more information on using ArrayFire on macOS, visit the following
[page](http://arrayfire.org/docs/using_on_osx.htm).

## NVIDIA Tegra devices

ArrayFire is capable of running TX2 devices.

Before installing ArrayFire, make sure the latest version of JetPack (v2.3 and
above) or L4T (v24.2 and above) is installed.

### Tegra prerequisites

The following dependencies are required for Tegra devices:

    sudo apt install libopenblas-dev liblapacke-dev

## Testing installation

After ArrayFire is finished installing, we recommend building and running a
few of the provided examples to verify things are working as expected.

On Windows, open the CMakeLists.txt file from CMake-GUI. Once the project is
configured and generated, build and run the examples from Visual Studio.

On Linux, run the following commands:

    cp -r /opt/arrayfire/share/ArrayFire/examples /tmp/examples
    cd /tmp/examples
    mkdir build
    cd build
    cmake ..
    make
    ./helloworld/helloworld_{cpu,cuda,oneapi,opencl}

## <a name="GettingHelp"></a> Getting help

* Google Groups: https://groups.google.com/forum/#!forum/arrayfire-users
* ArrayFire Services:  [Consulting](https://arrayfire.com/consulting/)  |  [Training](https://arrayfire.com/training/)
* ArrayFire Blogs: http://arrayfire.com/blog/
* Email: <mailto:support@arrayfire.com>
