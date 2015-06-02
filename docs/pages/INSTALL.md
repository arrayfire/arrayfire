ArrayFire binary installation instructions {#installing}
=====

Installing ArrayFire couldn't be easier. We ship installers for Windows,
OSX, and several variants of Linux. In general the installation procedure
proceeds like this:

1. [Download](http://arrayfire.com/download/) the ArrayFire installer for your
   operating system
2. Install prerequisites
3. Install ArrayFire
4. Test the installation
5. [Where to go for help?](#GettingHelp)

Below you will find instructions for

* [Windows](#Windows)
* Linux including
    * [Debian (.deb) 8](#Debian)
    * [Ubuntu (.deb) 14.10 and later](#Ubuntu)
    * [Fedora (.rpm) 21](#Fedora)
* [Mac OSX (.sh and brew)](#OSX)

# <a name="Windows"></a> Windows

Simply [download](http://arrayfire.com/download/) and run the installer.
If you wish to use CUDA or OpenCL please ensure that you have also installed
support for these technologies from your video card vendor's website.

# Linux

## <a name="Debian"></a> Debian 8

First [download](http://arrayfire.com/download/) ArrayFire. Then, using the
`gdebi` package manager, you can install ArrayFire and all dependencies as
follows:

    gdebi arrayfire*.deb

If you prefer to use the `.sh` installer, it and all prerequisite packages
may be installed as follows:

    # Prerequisite packages:
    apt-get install libfreeimage-dev libatlas3gf-base libfftw3-dev cmake

    # Enable GPU support (OpenCL):
    apt-get install ocl-icd-libopencl1

    # Run Installer
    ./arrayfire_3.0.0_Linux_x86_64.sh --exclude-subdir --prefix=/usr/local

To enable CUDA support, edit `/etc/apt/sources.list` and append `non-free`
to the line containing `deb http://.../debian jessie main`. Then, as root, run

    apt-get update
    apt-get install nvidia-cuda-dev

## <a name="Fedora"></a> Fedora 21

First [download](http://arrayfire.com/download/) ArrayFire. Then, using the
`yum` package manager, you can install ArrayFire and all dependencies as
follows:

    yum --nogpgcheck localinstall arrayfire*.rpm

Or with the self-extracting installer

    # Install prerequiste packages
    yum install freeimage atlas fftw cmake

    # Run Installer
    ./arrayfire_3.0.0_Linux_x86_64.sh --exclude-subdir --prefix=/usr/local

## <a name="Ubuntu"></a> Ubuntu 14.10 and later

First [download](http://arrayfire.com/download/) ArrayFire. Then, using the
`gdebi` package manager, you can install ArrayFire and all dependencies as
follows:

    sudo apt-get install gdebi
    gdebi arrayfire*.deb

If you prefer to use the `.sh` installer, it and all prerequisite packages
may be installed as follows:

    # Prerequisite packages:
    sudo apt-get install libfreeimage-dev libatlas3gf-base libfftw3-dev cmake

    # Enable GPU support (OpenCL and/or CUDA):
    sudo apt-get install ocl-icd-libopencl1
    sudo apt-get install nvidia-cuda-dev

    # Run Installer
    sudo ./arrayfire_3.0.0_Linux_x86_64.sh --exclude-subdir --prefix=/usr/local

# <a name="OSX"></a> Mac OSX

## Self-extracting zip from ArrayFire website

On OSX there are several dependencies that are not integrated into the
operating system. It is easiest to install these using [Homebrew](http://brew.sh/),
but you can also build them yourself if you prefer.

First [download](http://arrayfire.com/download/) ArrayFire. You may install
ArrayFire to `/usr/local` from XTerm using the following commands:

    brew install boost fftw cmake freeimage

    sudo ./arrayfire_3.0.0_Linux_x86_64.sh --exclude-subdir --prefix=/usr/local

## Brew installation

GitHub user [sutoiku](https://github.com/sutoiku) has been kind enough to
write a brew installation script for ArrayFire. This installation method will
download and compile ArrayFire and all prerequisites. Please remember to
register on the ArrayFire website so we can keep you up to date about new
versions of our software!

    brew install arrayfire

## Testing installation

After ArrayFire is installed, you can build the example programs as follows:

    cp -r /usr/local/share/doc/arrayfire/examples .
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
