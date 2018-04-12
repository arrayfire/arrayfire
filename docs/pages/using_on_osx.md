Using ArrayFire on OSX {#using_on_osx}
=====

Once you have [installed](\ref installing) ArrayFire on your system, the next thing to do is set up your build system. On OSX, you may create ArrayFire project using almost any editor, compiler, or build system. The only requirement is that you can include the ArrayFire header directory, and link with the ArrayFire library you intend to use.

## <a name="big-picture"/> The big picture

By default, the ArrayFire OSX installer will place several files in your computer's `/opt/arrayfire` directory. The installer will populate this directory with files in the following sub-directories:

    include/arrayfire.h         - Primary ArrayFire include file
    include/af/*.h              - Additional include files
    lib/libaf*                  - CPU, CUDA, and OpenCL libraries (.a, .so)
    lib/libforge*               - Visualization library
    lib/libcu*                  - CUDA backend dependencies
    lib/libglbinding*           - OpenGL graphics dependencies
    share/ArrayFire/cmake/*     - CMake config (find) scripts
    share/ArrayFire/examples/*  - All ArrayFire examples

Because ArrayFire follows standard installation practices, you can use basically any build system to create and compile projects that use ArrayFire. Among the many possible build systems on Linux we suggest using ArrayFire with either CMake or Makefiles with CMake being our preferred build system.

## Build Instructions:
* [CMake](#CMake)
* [Makefiles](#Makefiles)
* [XCode](#XCode)

## <a name="CMake"></a>CMake

We recommend that the CMake build system be used to create ArrayFire projects. As [discussed above](#big-picture), ArrayFire ships with a series of CMake scripts to make finding and using our library easy.

First create a file called `CMakeLists.txt` in your project directory:

    cd your-project-directory
    touch CMakeLists.txt

and populate it with the following code:

    find_package(ArrayFire)

    ... [gather source files, etc.]

    add_executable(my_executable [list your source files here])

    # To use Unified backend, do the following.
    # Unified backend lets you choose the backend at runtime
    target_link_libraries(my_executable ArrayFire::af)

where `my_executable` is the name of the executable you wish to create. See the [CMake documentation](https://cmake.org/documentation/) for more information on how to use CMake. To link with a specific backend directly, replace the `ArrayFire::af` with the following for their respective backends.

* `ArrayFire::afcpu` for CPU backend.
* `ArrayFire::afcuda` for CUDA backend.
* `ArrayFire::afopencl` for OpenCL backend.

Next we need to instruct CMake to create build instructions and then compile. We suggest using CMake's out-of-source build functionality to keep your build and source files cleanly separated. To do this open the CMake GUI.

    cd your-project-directory
    mkdir build
    cd build
    cmake ..
    make

*NOTE:* If you have installed ArrayFire to a non-standard location, CMake can still help you out. When you execute CMake specify the path to ArrayFire installation root as `ArrayFire_DIR` variable.

For example, if ArrayFire were installed locally to `/home/user/ArrayFire` then you would modify the `cmake` command above to contain the following definition:

    cmake -DArrayFire_DIR=/home/user/ArrayFire ..

You can also specify this information in the `ccmake` command-line interface.

## <a name="Makefiles"></a> Makefiles

Building ArrayFire projects with Makefiles is fairly similar to CMake except you must specify all paths and libraries manually.

As with any make project, you need to specify the include path to the directory containing `arrayfire.h` file. This should be `-I /opt/arrayfire/include` if you followed our installation instructions.

Similarly, you will need to specify the path to the ArrayFire library using the `-L` option (e.g. `-L/opt/arrayfire/lib`) followed by the specific ArrayFire library you wish to use using the `-l` option (for example `-lafcpu`, `-lafopencl`, `-lafcuda`, or `-laf` for the CPU, OpenCL, CUDA, and unified backends respectively.

Here is a minimal example Makefile which uses ArrayFire's CPU backend:

    LIBS=-lafcpu
    LIB_PATHS=-L/opt/arrayfire/lib
    INCLUDES=-I/opt/arrayfire/include
    CC=g++ $(COMPILER_OPTIONS)
    COMPILER_OPTIONS=-std=c++11 -g

    all: main.cpp Makefile
        $(CC) main.cpp -o test $(INCLUDES) $(LIBS) $(LIB_PATHS)

## <a name="XCode"></a> XCode

Although we recommend using CMake to build ArrayFire projects on OSX, you can use XCode if this is your preferred development platform. To save some time, we have created an sample XCode project in our [ArrayFire Project Templates repository](https://github.com/arrayfire/arrayfire-project-templates).

To set up a basic C/C++ project in XCode do the following:

1. Start up XCode. Choose macOS -> Application, Command Line Tool for the project:
\htmlonly
<br />
<a href="xcode-setup/xcode-startup.png">
<img src="xcode-setup/xcode-startup.png" alt="Create a command line tool XCode Project" align="middle" width="50%" />
</a>
\endhtmlonly

2. Fill in the details for your project and choose either C or C++ for the project:
\htmlonly
<br />
<a href="xcode-setup/project-options.png">
<img src="xcode-setup/project-options.png" alt="Create a C/C++ project" align="middle" width="50%" />
</a>
\endhtmlonly

3. Next we need to configure the build settings. In the left-hand pane, click    on the project. In the center pane, click on "Build Settings" followed by   the "All" button:
\htmlonly
<br />
<a href="xcode-setup/build-settings.png">
<img src="xcode-setup/build-settings.png" alt="Configure build settings" align="middle" width="50%" />
</a>
\endhtmlonly

4. Now search for "Header Search Paths" and add `/opt/arrayfire/include` to the list:
\htmlonly
<br />
<a href="xcode-setup/header-search-paths.png">
<img src="xcode-setup/header-search-paths.png" alt="Configure build settings" align="middle" width="50%" />
</a>
\endhtmlonly

5. Then search for "Library Search Paths" and add `/opt/arrayfire/lib` to the list:
\htmlonly
<br />
<a href="xcode-setup/library-search-paths.png">
<img src="xcode-setup/library-search-paths.png" alt="Configure build settings" align="middle" width="50%" />
</a>
\endhtmlonly

6. Next, we need to make sure the executable is linked with an ArrayFire library:
   To do this, click the "Build Phases" tab and expand the "Link with Binary Library"
   menu:
\htmlonly
<br />
<a href="xcode-setup/build-phases.png">
<img src="xcode-setup/build-phases.png" alt="Configure build settings" align="middle" width="50%" />
</a>
\endhtmlonly

7. In the search dialog that pops up, choose the "Add Other" button from the lower right. Specify the `/opt/arrayfire/lib` folder:
\htmlonly
<br />
<a href="xcode-setup/library-folder-path.png">
<img src="xcode-setup/library-folder-path.png" alt="Configure build settings" align="middle" width="50%" />
</a>
\endhtmlonly

8. Lastly, select the ArrayFire library with which you wish to link your program. Your options will be:
~~~~~
libafcuda.*.dylib   - CUDA backend
libafopencl.*.dylib - OpenCL backend
libafcpu.*.dylib    - CPU backend
libaf.*.dylib       - Unified backend
~~~~~
In the picture below, we have elected to link with the OpenCL backend:
\htmlonly
<br />
<a href="xcode-setup/pick-arrayfire-library.png">
<img src="xcode-setup/pick-arrayfire-library.png" alt="Configure build settings" align="middle" width="50%" />
</a>
\endhtmlonly

9. Lastly, lets test ArrayFire's functionality. In the left hand pane open the main.cpp` file and insert the following code:

~~~~~
// Include the ArrayFire header file
#include <arrayfire.h>

int main(int argc, const char * argv[]) {
    // Gather some information about the ArrayFire device
    af::info();
    return 0;
}
~~~~~

Finally, click the build button and you should see some information about your graphics card in the lower-section of your screen:

\htmlonly
<br />
<a href="xcode-setup/afinfo-result.png">
<img src="xcode-setup/afinfo-result.png" alt="Configure build settings" align="middle" width="50%" />
</a>
\endhtmlonly
